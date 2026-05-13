"""
_03_near_deduplication.py

Tree-sitter-based, parallel, authorship-safe near-deduplication:

1. Tree-sitter parse (Python / Java / C++) in PARALLEL
2. Canonicalize *local* identifiers (params, locals, loop vars)
3. Serialize syntax tree with semantic anchors
4. Shingle (n-grams) + MinHash
5. LSH retrieval + verified similarity graph
6. Connected-components clustering
7. Deterministic survivor selection
8. Persist near-duplicate cluster metadata
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import builtins
import hashlib
import re
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from tree_sitter import Parser

from .utils.resource_manager import ResourceManager
from .utils.languages import get_language

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

NUM_HASHES = 128
LSH_THRESHOLD = 0.85
SHINGLE_N = 5
N_JOBS = -1  # use all cores

BUILTINS = set(dir(builtins))
TOK_RE = re.compile(r"\w+|[^\s\w]")

MAX_WORKERS = ResourceManager.get_cpu_limit()

# ----------------------------
# TREE-SITTER HELPERS
# ----------------------------

PARAM_PARENT_TYPES = {
    "parameter",
    "parameter_declaration",
    "formal_parameter",
    "formal_parameters",
    "parameter_list",
}

LOCAL_DECL_PARENT_TYPES = {
    "local_variable_declaration",
    "variable_declarator",
    "init_declarator",
    "for_statement",
    "enhanced_for_statement",
    "for_in_statement",
    "for_range_loop",
}

LITERAL_NODE_TYPES = {
    "string_literal",
    "character_literal",
    "number_literal",
    "integer_literal",
    "float_literal",
    "true",
    "false",
    "null",
}


def serialize_tree_sitter(root_node, code_bytes, lang: str):
    tokens = []

    local_names = set()
    var_map = {}
    arg_map = {}
    var_counter = 0
    arg_counter = 0

    def get_text(node):
        return code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def collect_locals(node, parent_type=None):
        nonlocal local_names
        if parent_type in PARAM_PARENT_TYPES and node.type == "identifier":
            local_names.add(get_text(node))
        if parent_type in LOCAL_DECL_PARENT_TYPES and node.type == "identifier":
            local_names.add(get_text(node))
        for child in node.children:
            collect_locals(child, node.type)

    collect_locals(root_node, None)

    def canon_var(name):
        nonlocal var_counter
        if name not in var_map:
            var_counter += 1
            var_map[name] = f"v{var_counter}"
        return var_map[name]

    def canon_arg(name):
        nonlocal arg_counter
        if name not in arg_map:
            arg_counter += 1
            arg_map[name] = f"p{arg_counter}"
        return arg_map[name]

    def walk(node, parent_type=None):
        tokens.append(node.type)

        if node.type == "identifier":
            name = get_text(node)
            if lang == "python" and name in BUILTINS:
                tokens.append(f"ID:{name}")
            elif name in local_names:
                if parent_type in PARAM_PARENT_TYPES:
                    tokens.append(f"ID:{canon_arg(name)}")
                else:
                    tokens.append(f"ID:{canon_var(name)}")
            else:
                tokens.append(f"ID:{name}")

        elif node.type in LITERAL_NODE_TYPES:
            text = get_text(node)
            if text in {"true", "false", "null"}:
                tokens.append(f"CONST:{text}")
            else:
                try:
                    v = int(text)
                    tokens.append(f"SMALLINT:{v}" if abs(v) <= 10 else "INT")
                except ValueError:
                    try:
                        v = float(text)
                        tokens.append(f"SMALLFLOAT:{v}" if abs(v) <= 10 else "FLOAT")
                    except ValueError:
                        tokens.append(f"SMALLSTR:{text}" if len(text) <= 8 else "STR")

        for child in node.children:
            walk(child, node.type)

    walk(root_node, None)
    return tokens


# ----------------------------
# MINHASH
# ----------------------------


def make_ngrams(seq, n=SHINGLE_N):
    if len(seq) < n:
        return [" ".join(seq)]
    return [" ".join(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def minhash_from_shingles(shingles):
    m = MinHash(num_perm=NUM_HASHES, seed=SEED)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


# ----------------------------
# SURVIVOR PRIORITY
# ----------------------------


def sample_priority(row):
    return (
        row.get("source_quality", 0),
        len(row.get("code", "")),
        -row["_orig_idx"],
    )


# ----------------------------
# PARALLEL WORKER
# ----------------------------


def process_sample(i, row):
    code = row["code"]
    lang_raw = str(row.get("language", "python")).lower()

    # map to TS languages
    if lang_raw.startswith("py"):
        ts_lang = "python"
    elif lang_raw.startswith("java") and "script" not in lang_raw:
        ts_lang = "java"
    elif lang_raw in {"c++", "cpp", "cxx"}:
        ts_lang = "cpp"
    else:
        tokens = TOK_RE.findall(code)
        structural_repr = " ".join(tokens)
        shingles = make_ngrams(tokens, SHINGLE_N)
        mh = minhash_from_shingles(shingles)
        return i, mh, hashlib.sha256(structural_repr.encode()).hexdigest()

    try:
        parser = Parser()
        parser.language = get_language(ts_lang)
        code_bytes = code.encode("utf-8", errors="ignore")
        tree = parser.parse(code_bytes)
        tokens = serialize_tree_sitter(tree.root_node, code_bytes, ts_lang)
    except Exception:
        tokens = TOK_RE.findall(code)

    structural_repr = " ".join(tokens)
    shingles = make_ngrams(tokens, SHINGLE_N)
    mh = minhash_from_shingles(shingles)

    return i, mh, hashlib.sha256(structural_repr.encode()).hexdigest()


# ----------------------------
# GRAPH / CONNECTED COMPONENTS
# ----------------------------


def connected_components(adj):
    visited = set()
    components = []
    for node in adj:
        if node in visited:
            continue
        comp = []
        queue = deque([node])
        visited.add(node)
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        components.append(comp)
    return components


# ----------------------------
# MAIN PIPELINE
# ----------------------------


def run_step_03():
    files = list(INPUT_DIR.glob("*.parquet"))
    if not files:
        print("No files found.")
        return

    dfs = {f.stem: pd.read_parquet(f) for f in files}
    initial_counts = Counter({k: len(v) for k, v in dfs.items()})

    # Flatten rows
    rows = []
    print("Flattening datasets...")
    for ds_name, df in dfs.items():
        for r in tqdm(df.itertuples(index=True), total=len(df), desc=f"{ds_name}"):
            row = r._asdict()  # type: ignore
            row["_orig_ds"] = ds_name
            row["_orig_idx"] = r.Index
            row["near_dup_cluster_id"] = None
            row["near_dup_cluster_size"] = 1
            row["canonical_survivor"] = False
            row["structural_hash"] = None
            rows.append(row)

    print(f"Total samples before near-dedup: {len(rows)}")

    # ----------------------------
    # Build MinHash + LSH index
    # ----------------------------
    print("Parallel parsing + MinHash...")
    import sys

    sys.stdout.flush()

    results: list[Any] = [None] * len(rows)
    print(f"Processing {len(rows)} samples with {MAX_WORKERS} workers...")
    sys.stdout.flush()

    raw_chunks = max(1, len(rows) // (max(1, MAX_WORKERS) * 16))
    chunksize = max(1, min(1024, raw_chunks))
    print(f"Using chunksize={chunksize} for executor.map (raw={raw_chunks})")
    sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        mapped = ex.map(process_sample, range(len(rows)), rows, chunksize=chunksize)
        it = iter(mapped)

        for _ in tqdm(range(len(rows)), desc="Parsing + MinHash (parallel)"):
            i, mh, structural_hash = next(it)
            results[i] = (mh, structural_hash)

    # Fill any missing results with a fallback
    for idx, val in enumerate(results):
        if val is None:
            results[idx] = (MinHash(num_perm=NUM_HASHES, seed=SEED), None)

    print("Building signatures...")

    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_HASHES)
    signatures = {}
    id_to_idx = {}

    for i, (mh, structural_hash) in tqdm(
        enumerate(results), total=len(results), desc="Building signatures"
    ):
        rows[i]["structural_hash"] = structural_hash
        sample_id = f"{rows[i]['_orig_ds']}::{rows[i]['_orig_idx']}"
        signatures[sample_id] = mh
        id_to_idx[sample_id] = i
        lsh.insert(sample_id, mh)

    # ----------------------------
    # Build similarity graph via LSH (with verification)
    # ----------------------------
    print("Building similarity graph...")
    adj = defaultdict(set)
    for sample_id, mh in tqdm(signatures.items(), desc="LSH queries"):
        for nb in lsh.query(mh):
            if nb == sample_id:
                continue
            if mh.jaccard(signatures[nb]) >= LSH_THRESHOLD:
                adj[sample_id].add(nb)
                adj[nb].add(sample_id)

    # ----------------------------
    # Connected components = near-dup clusters
    # ----------------------------
    print("Computing connected components...")
    components = connected_components(adj)
    print(f"Found {len(components)} clusters.")

    keep_mask = np.ones(len(rows), dtype=bool)
    cluster_id_counter = 0
    pair_records = []

    for comp in tqdm(components, desc="Processing clusters"):
        cluster_label = f"cluster_{cluster_id_counter}"
        cluster_id_counter += 1

        cluster_rows = [rows[id_to_idx[sid]] for sid in comp]
        cluster_size = len(cluster_rows)

        for r in cluster_rows:
            r["near_dup_cluster_id"] = cluster_label
            r["near_dup_cluster_size"] = cluster_size

        if cluster_size == 1:
            cluster_rows[0]["canonical_survivor"] = True
            continue

        survivor_row = max(cluster_rows, key=sample_priority)
        survivor_sid = f"{survivor_row['_orig_ds']}::{survivor_row['_orig_idx']}"
        survivor_idx = id_to_idx[survivor_sid]
        rows[survivor_idx]["canonical_survivor"] = True

        for sid in comp:
            idx = id_to_idx[sid]
            if idx != survivor_idx:
                keep_mask[idx] = False
                pair_records.append(
                    {
                        "pair_id": len(pair_records),
                        "pair_type": "near",
                        "cluster_id": cluster_label,
                        "cluster_size": cluster_size,
                        "group_id": cluster_label,
                        "group_size": cluster_size,
                        "removed_sample_id": sid,
                        "removed_dataset": rows[idx]["_orig_ds"],
                        "removed_idx": rows[idx]["_orig_idx"],
                        "removed_row_index": idx,
                        "survivor_sample_id": survivor_sid,
                        "survivor_dataset": survivor_row["_orig_ds"],
                        "survivor_idx": survivor_row["_orig_idx"],
                        "survivor_row_index": survivor_idx,
                        "jaccard_to_survivor": signatures[sid].jaccard(signatures[survivor_sid]),
                        "similarity": signatures[sid].jaccard(signatures[survivor_sid]),
                    }
                )

    # ----------------------------
    # Apply mask and save
    # ----------------------------
    final_rows = [r for r, keep in zip(rows, keep_mask) if keep]
    print(f"Final survivors: {len(final_rows)}")

    out_map = defaultdict(list)
    for r in final_rows:
        out_map[r["_orig_ds"]].append(r)

    final_counts = Counter()
    for ds_name, items in tqdm(out_map.items(), desc="Saving datasets"):
        df = pd.DataFrame(items)
        df = df.drop(columns=["_orig_ds", "_orig_idx"], errors="ignore")
        df.to_parquet(OUTPUT_DIR / f"{ds_name}.parquet", index=False)
        final_counts[ds_name] = len(df)

    pair_df = pd.DataFrame(pair_records)
    pair_df.to_parquet(REPORT_DIR / "_03_near_duplicate_pairs.parquet", index=False)
    pair_df.to_csv(REPORT_DIR / "_03_near_duplicate_pairs.csv", index=False)

    # ----------------------------
    # Report
    # ----------------------------
    with open(REPORT_DIR / "_03_near_deduplication_report.txt", "w") as f:
        f.write("=== Parallel Tree-sitter + MinHash NEAR-DEDUP REPORT ===\n\n")
        for ds in sorted(initial_counts.keys()):
            start = initial_counts[ds]
            end = final_counts.get(ds, 0)
            reduction = 100 * (1 - end / start) if start > 0 else 0
            f.write(f"{ds}: {start} -> {end} ({reduction:.2f}% reduction)\n")

        f.write("\nPair lookup files:\n")
        f.write(f"- {REPORT_DIR / '_03_near_duplicate_pairs.parquet'}\n")
        f.write(f"- {REPORT_DIR / '_03_near_duplicate_pairs.csv'}\n")

    print("Step 03 complete.")
