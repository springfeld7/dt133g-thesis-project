"""_01_deduplicate.py

Step 01: Semantic Cleaning Layer (VarCLR-based Deduplication with HNSW)
"""

from pathlib import Path
from collections import Counter, defaultdict
import random
import hashlib

import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm

from evaluation.varclr.models.encoders import BERT

# ----------------------------
# CONFIG
# ----------------------------

INPUT_DIR = Path("data/_00_normalized_datasets")
OUTPUT_DIR = Path("data/_01_varclr_cleaned")
REPORT_DIR = Path("output/_01_varclr_reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 256
KNN = 10
SIM_THRESHOLD = 0.90


# ----------------------------
# UNION FIND
# ----------------------------

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


# ----------------------------
# GLOBAL EXACT DEDUP
# ----------------------------

def global_dedup(files: list[Path]) -> list[dict]:
    """
    Deduplicates across datasets using MD5 hash.
    Keeps one random representative per hash group to ensure cross-dataset fairness.

    Args:
        files (list[Path]): List of paths to parquet files.

    Returns:
        list[dict]: A list of deduplicated record dictionaries.
    """
    groups = defaultdict(list)

    for f in files:
        df = pd.read_parquet(f)
        df["source_dataset"] = f.stem

        for r in df.to_dict("records"):
            h = r.get("code_hash") or hashlib.md5(r["code"].encode()).hexdigest()
            groups[h].append(r)

    kept = []
    for group in groups.values():
        kept.append(random.choice(group))

    return kept


# ----------------------------
# EMBEDDING
# ----------------------------

def embed_batch(model: torch.nn.Module, texts: list[str], device: torch.device) -> np.ndarray:
    """
    GPU-safe batched encoding for semantic representation.

    Args:
        model (torch.nn.Module): The VarCLR/BERT encoder model.
        texts (list[str]): A batch of code strings to encode.
        device (torch.device): The torch device (cuda/cpu).

    Returns:
        np.ndarray: A float32 numpy array of embeddings.
    """
    model.eval()
    with torch.no_grad():
        emb = model.encode(texts)   # <-- THIS is the correct path
        return emb.detach().cpu().numpy().astype("float32")

# ----------------------------
# MAIN PIPELINE
# ----------------------------

def run_step_01():
    """
    Executes the VarCLR + HNSW semantic deduplication pipeline.

    This includes:
    1. Exact deduplication via MD5.
    2. GPU-accelerated embedding generation.
    3. HNSW indexing for Approximate Nearest Neighbor (ANN) search.
    4. Disjoint Set Union (DSU) clustering of near-duplicates.
    5. Proportional fair selection to protect smaller datasets.
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

    # ----------------------------
    # LOAD INPUT FILES (multi-dataset parquet inputs)
    # ----------------------------
    files = list(INPUT_DIR.glob("*.parquet"))
    if not files:
        print("No files found in input directory.")
        return

    # ----------------------------
    # COMPUTE DATASET SIZE WEIGHTS
    # Used later for fairness-aware sampling in duplicate clusters
    # Larger datasets get lower weight, smaller datasets get higher survival chance
    # ----------------------------
    dataset_sizes = {}
    for f in files:
        df = pd.read_parquet(f)
        dataset_sizes[f.stem] = len(df)

    # ----------------------------
    # GLOBAL EXACT DEDUP (MD5 HASH)
    # Removes identical samples across ALL datasets before semantic stage
    # This reduces embedding + FAISS workload significantly
    # ----------------------------
    rows = global_dedup(files)
    print(f"After exact dedup: {len(rows)}")

    # ----------------------------
    # TRACK DATASET DISTRIBUTION
    # ----------------------------

    dataset_counts = Counter(r["source_dataset"] for r in rows)

    # This will be dynamically updated during cluster selection
    dataset_quota = {
        ds: count
        for ds, count in dataset_counts.items()
    }

    texts = [r["code"] for r in rows]

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

        batch = texts[i:i + BATCH_SIZE]

        # GPU forward pass → embeddings
        emb = embed_batch(model, batch, device)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(emb)

        # Initialize FAISS index once we know embedding dimension
        if index is None:
            dim = emb.shape[1]
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128

        index.add(emb)

    # ----------------------------
    # NEAREST NEIGHBOR SEARCH
    # WARNING: reconstruct_n() scales with index size
    # For very large datasets, consider batched search instead
    # ----------------------------
    print("Running ANN search...")
    D, I = index.search(index.reconstruct_n(0, index.ntotal), KNN)

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

    final_rows = []

    # ----------------------------
    # FAIR REPRESENTATIVE SELECTION PER CLUSTER
    # Instead of random removal, we bias survival toward smaller datasets
    # This prevents large datasets from dominating dedup output
    # ----------------------------
    for cluster_indices in tqdm(clusters.values(), desc="Reducing Clusters"):

        candidates = [rows[idx] for idx in cluster_indices]

        if len(candidates) == 1:
            final_rows.append(candidates[0])
            continue

        # ----------------------------
        # SMART SELECTION: protect small datasets globally
        # ----------------------------
        def score(c):
            ds = c["source_dataset"]
            
            # how "damaged" this dataset already is
            used = dataset_counts[ds] - dataset_quota[ds]
            
            # penalty grows as dataset loses samples
            penalty = used / (dataset_counts[ds] + 1e-9)

            return 1.0 / (1.0 + penalty)

        winner = max(candidates, key=score)

        final_rows.append(winner)

        # update quota bookkeeping
        dataset_counts[winner["source_dataset"]] -= 1

    initial_counts = dataset_sizes

    final_counts = Counter(r["source_dataset"] for r in final_rows)

    removed_counts = {
        ds: initial_counts.get(ds, 0) - final_counts.get(ds, 0)
        for ds in initial_counts
    }

    # ----------------------------
    # SAVE FINAL CLEANED DATASET
    # ----------------------------
    out_path = OUTPUT_DIR / "varclr_cleaned.parquet"
    pd.DataFrame(final_rows).to_parquet(out_path, index=False)


    # ----------------------------
    # SAVE DEDUPLICATION REPORT
    # ----------------------------
    report_path = REPORT_DIR / "step01_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== PROPORTIONAL FAIRNESS REPORT ===\n")
        f.write(f"Input samples (post-exact): {len(rows)}\n")
        f.write(f"Clusters found: {len(clusters)}\n")
        f.write(f"Final samples: {len(final_rows)}\n")
        f.write(f"Reduction: {100 * (1 - len(final_rows)/len(rows)):.2f}%\n\n")

        f.write("=== DATASET LOSS BREAKDOWN ===\n")
        for ds in initial_counts:
            start = initial_counts[ds]
            end = final_counts.get(ds, 0)
            removed = removed_counts[ds]

            f.write(f"{ds}:\n")
            f.write(f"  start:   {start}\n")
            f.write(f"  kept:    {end}\n")
            f.write(f"  removed: {removed}\n\n")

    print(f"Pipeline complete. Saved {len(final_rows)} samples.")


if __name__ == "__main__":
    run_step_01()
