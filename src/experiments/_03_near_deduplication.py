"""_03_near_deduplication.py

Step 03: Semantic Cleaning Layer (VarCLR-based Deduplication with HNSW)
"""

from pathlib import Path
from collections import Counter, defaultdict
import random

import numpy as np
import pandas as pd
import re
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluation.varclr.models.encoders import BERT
from evaluation.varclr.models import urls_pretrained_model
from .utils.calculate_balance_score import calculate_balance_score


# ----------------------------
# CONFIG
# ----------------------------

INPUT_DIR = Path("data/_02_exact_deduplicated_datasets")
OUTPUT_DIR = Path("data/_03_near_deduplicated_datasets")
REPORT_DIR = Path("output/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 4096
KNN = 10
SIM_THRESHOLD = 0.85
MAX_TOKENS = 510  # We reserve 2 tokens for special tokens of the model.
STRIDE = 256

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128


class DSU:
    """
    Union-Find structure for clustering near-duplicates.
    """

    def __init__(self, n: int):
        """
        Initializes the DSU with n elements.

        Args:
            n (int): The number of elements in the set.
        """
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """
        Finds the representative of the set containing x with path compression.

        Args:
            x (int): The element to find.

        Returns:
            int: The representative of the set.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        """
        Unites the sets containing a and b.

        Args:
            a (int): The first element.
            b (int): The second element.
        """
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def chunk_text(text: str, tokenizer, max_tokens: int = 510, stride: int = 256) -> list[str]:
    """
    Splits long code into overlapping token chunks.

    Args:
        text (str): Raw source code.
        tokenizer: HuggingFace tokenizer.
        max_tokens (int): Max tokens per chunk.
        stride (int): Overlap between chunks.

    Returns:
        list[str]: List of decoded text chunks.
    """
    # Manually apply the model's internal preprocessing first
    text = text.replace("@", "")
    text = re.sub("([a-z]|^)([A-Z]{1})", r"\1_\2", text).lower().replace("_", " ").strip()

    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))

        if end >= len(tokens):
            break

        start += max_tokens - stride

    return chunks


def embed_batch(model: torch.nn.Module, texts: list[str], tokenizer: AutoTokenizer) -> np.ndarray:
    """
    Flattens all chunks from the text batch into a single GPU pass.

    Args:
        model: VarCLR encoder with `.encode()`.
        texts (list[str]): Raw code samples.
        tokenizer: HuggingFace tokenizer used for chunking.
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Shape (batch_size, embedding_dim), float32 embeddings.
    """
    all_chunks = []
    chunk_counts = []  # To track how many chunks belong to each original text

    # Prepare all chunks
    for text in texts:
        chunks = chunk_text(text, tokenizer, max_tokens=MAX_TOKENS, stride=STRIDE)
        all_chunks.extend(chunks)
        chunk_counts.append(len(chunks))

    # Process all chunks in one go to leverage GPU parallelism
    if not all_chunks:
        return np.array([])

    # We use a sub-batch size here to avoid OOM if a file batch has
    # an insane amount of chunks. 128-256 is usually safe for 8GB.
    gpu_sub_batch = 128
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(all_chunks), gpu_sub_batch):
            batch_slice = all_chunks[i : i + gpu_sub_batch]
            embs = model.encode(batch_slice)
            all_embeddings.append(embs)

    flattened_embs = torch.cat(all_embeddings)

    # --- Mean Pooling Phase: Re-aggregate chunks back to original files ---
    final_embeddings = []
    cursor = 0
    for count in chunk_counts:
        file_chunks = flattened_embs[cursor : cursor + count]
        final_embeddings.append(file_chunks.mean(dim=0))
        cursor += count

    return torch.stack(final_embeddings).cpu().numpy().astype("float32")


# ----------------------------
# MAIN PIPELINE
# ----------------------------


def run_step_03():
    """
    Executes the VarCLR + HNSW semantic deduplication pipeline.

    This includes:
    1. Embedding generation.
    2. HNSW indexing for Approximate Nearest Neighbor (ANN) search.
    3. Disjoint Set Union (DSU) clustering of near-duplicates.
    4. Proportional fair selection to protect smaller datasets.
    5.
    """

    # ----------------------------
    # DEVICE SELECTION (GPU preferred, CPU fallback)
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # LOAD VARCLR ENCODER
    # IMPORTANT: model.encode() already handles tokenization + forward pass
    # ----------------------------
    model = BERT.from_pretrained("varclr-codebert")
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(urls_pretrained_model.PRETRAINED_TOKENIZER)

    # ----------------------------
    # LOAD INPUT FILES (multi-dataset parquet inputs)
    # ----------------------------
    files = list(INPUT_DIR.glob("*.parquet"))
    if not files:
        print("No files found in input directory.")
        return

    print("Loading datasets...")
    dfs = {f.stem: pd.read_parquet(f) for f in files}

    print("Loading datasets (testing with first 500 rows)...")

    initial_counts = Counter({k: len(v) for k, v in dfs.items()})

    # Flatten all samples into a single list with metadata for tracking
    rows = []
    for ds_name, df in dfs.items():
        for idx, r in df.iterrows():
            row_data = r.to_dict()
            row_data["_orig_ds"] = ds_name
            row_data["_orig_idx"] = idx
            rows.append(row_data)

    print(f"Input to semantic stage: {len(rows)}")

    texts = [r["code_normalized"] for r in rows]

    # ----------------------------
    # FAISS INDEX INITIALIZATION (lazy init)
    # We only create index after first embedding batch to infer dimension
    # ----------------------------
    dim = None
    index = None

    print("Building FAISS index...")

    # ----------------------------
    # EMBEDDING + INDEX BUILD (STREAMING)
    # This avoids storing all embeddings in memory at once
    # Critical for scaling to ~1M+ samples
    # ----------------------------
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding+Indexing"):
        batch = texts[i : i + BATCH_SIZE]

        # GPU forward pass → embeddings
        emb = embed_batch(model, batch, tokenizer)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(emb)

        # Initialize FAISS index once we know embedding dimension
        if index is None:
            dim = emb.shape[1]
            index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            index.hnsw.efSearch = HNSW_EF_SEARCH

        index.add(emb)

    # ----------------------------
    # NEAREST NEIGHBOR SEARCH
    # WARNING: reconstruct_n() scales with index size
    # For very large datasets, consider batched search instead
    # ----------------------------
    print("Running ANN search...")
    D_list, I_list = [], []
    batch_size_search = 100000

    for i in tqdm(range(0, index.ntotal, batch_size_search), desc="Searching"):
        # Pull only a small chunk of vectors to query
        n_to_extract = min(batch_size_search, index.ntotal - i)
        query_vectors = index.reconstruct_n(i, n_to_extract)

        # Search for neighbors for this chunk
        batch_d, batch_i = index.search(query_vectors, KNN)
        D_list.append(batch_d)
        I_list.append(batch_i)

    # Stack results back into full arrays
    D = np.vstack(D_list)
    I = np.vstack(I_list)

    # ----------------------------
    # DSU CLUSTERING
    # Groups semantically similar samples into connected components
    # Each cluster represents a "near-duplicate group"
    # ----------------------------
    dsu = DSU(len(rows))

    for i in range(len(rows)):
        for neighbor_idx, sim in zip(I[i], D[i]):
            if neighbor_idx != i and neighbor_idx != -1:
                if sim >= SIM_THRESHOLD:
                    dsu.union(i, neighbor_idx)

    # ----------------------------
    # BUILD CLUSTERS FROM DSU STRUCTURE
    # Each root node becomes a cluster
    # ----------------------------
    clusters = defaultdict(list)
    for i in range(len(rows)):
        clusters[dsu.find(i)].append(i)

    # We reset the counts before this stage to treat the post-exact pool
    # as the current 'truth' for the semantic stage.
    current_pool_counts = Counter(r["_orig_ds"] for r in rows)
    remaining_in_stage = dict(current_pool_counts)

    drop_indices = defaultdict(set)

    # ----------------------------
    # FAIR REPRESENTATIVE SELECTION PER CLUSTER
    # ----------------------------
    for cluster_indices in tqdm(clusters.values(), desc="Reducing Clusters"):
        candidates = [rows[idx] for idx in cluster_indices]

        if len(candidates) == 1:
            continue

        # Pick the sample from the dataset that has the highest % of its data remaining.
        winner = max(
            candidates,
            key=lambda iter_var: calculate_balance_score(
                iter_var["_orig_ds"], remaining_in_stage, current_pool_counts
            ),
        )

        # Mark losers for drop and update stage health
        for c in candidates:
            if c["_orig_ds"] == winner["_orig_ds"] and c["_orig_idx"] == winner["_orig_idx"]:
                continue

            drop_indices[c["_orig_ds"]].add(c["_orig_idx"])
            remaining_in_stage[c["_orig_ds"]] -= 1

    # SAVE & REPORT
    final_counts = Counter()
    for ds_name, df in dfs.items():
        indices_to_drop = drop_indices[ds_name]
        cleaned_df = df.drop(index=list(indices_to_drop)).reset_index(drop=True)
        cleaned_df = cleaned_df.drop(
            columns=["code_normalized", "normalized_hash"], errors="ignore"
        )

        out_path = OUTPUT_DIR / f"{ds_name}.parquet"
        cleaned_df.to_parquet(out_path, index=False)
        final_counts[ds_name] = len(cleaned_df)

    report_path = REPORT_DIR / "_03_near_deduplication_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== UNIFIED FAIRNESS REPORT ===\n")
        for ds in initial_counts:
            start = initial_counts[ds]
            end = final_counts.get(ds, 0)
            f.write(f"{ds}: {start} -> {end} (Loss: {100*(1-end/start):.2f}%)\n")

    print(f"Pipeline complete. Processed {len(initial_counts)} separate files.")


if __name__ == "__main__":
    run_step_03()
