"""_01_extract_valid_samples.py

Step one of experiments: Extract the valid samples from the DroidCollection dataset.
Valid samples are defined as those that:

    - Belong to one of the following languages: Python, Java, or C++
    - Have a label of either MACHINE_GENERATED or HUMAN_GENERATED
    - Are successfully parsed by TranStructiver without any ERROR nodes in the resulting parse tree
"""

import hashlib
from pathlib import Path
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .sample_selection.dataset_manager import DatasetManager
from .sample_selection.analysis import SampleAnalyzer


BATCH_SIZE = 10_000


def run_step_01():
    """
    Step 01: Initial Filtering, Metric Extraction, and Persistent Storage.

    1. Streams data and filters for valid parse trees.
    2. Extracts stylistic metrics.
    3. Deduplicates via MD5 hashing.
    4. Writes dataset incrementally to Parquet.
    5. Computes summary statistics in a streaming fashion.
    6. Saves summary report to output/dataset_summary.txt.
    """

    manager = DatasetManager()
    analyzer = SampleAnalyzer()

    manager.authenticate()
    stream = manager.get_iterator()

    print("\n[Dataset Shape/Schema]")
    print(f"Columns: {list(stream.features.keys())}")
    total_rows = stream.info.splits["train"].num_examples
    print(f"Total Rows (Metadata): {total_rows}")

    print("--- Step 01: Streaming & Initial Filtering ---")

    output_path = Path("data/valid_samples/valid_samples.parquet")
    report_path = Path("output/dataset_summary.txt")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None

    seen_hashes = set()
    batch = []

    # ----------------------------
    # STREAMING STATS ACCUMULATORS
    # ----------------------------

    numeric_cols = [
        "char_count",
        "loc",
        "lloc",
        "identifier_density",
        "for_loop_density",
        "comment_density",
        "whitespace_ratio",
    ]

    global_sum = defaultdict(float)
    global_count = 0

    lang_sum = defaultdict(lambda: defaultdict(float))
    lang_count = defaultdict(int)

    granular_sum = defaultdict(lambda: defaultdict(float))
    granular_count = defaultdict(int)

    # ----------------------------
    # STREAM LOOP
    # ----------------------------
    for entry in stream:

        lang = entry.get("Language")
        label = entry.get("Label")
        code = entry.get("Code", "")

        valid_tree = analyzer.get_valid_tree(code, lang, label)
        if not valid_tree:
            continue

        code_hash = hashlib.md5(code.encode("utf-8")).hexdigest()
        if code_hash in seen_hashes:
            continue
        seen_hashes.add(code_hash)

        row_data = analyzer.calculate_metrics(code, lang, label, valid_tree)

        row_data["code"] = code
        row_data["code_hash"] = code_hash
        row_data["language"] = lang
        row_data["label"] = label

        # ----------------------------
        # UPDATE STATS (ONLINE)
        # ----------------------------
        global_count += 1

        for col in numeric_cols:
            val = row_data[col]
            global_sum[col] += val
            lang_sum[lang][col] += val
            granular_sum[(lang, label)][col] += val

        lang_count[lang] += 1
        granular_count[(lang, label)] += 1

        # ----------------------------
        # BATCH WRITE
        # ----------------------------
        batch.append(row_data)

        if len(batch) >= BATCH_SIZE:
            df = pd.DataFrame(batch)
            table = pa.Table.from_pandas(df)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)

            print(f"Wrote {len(batch)} rows...")

            batch.clear()

    # ----------------------------
    # FINAL FLUSH
    # ----------------------------
    if batch:
        df = pd.DataFrame(batch)
        table = pa.Table.from_pandas(df)

        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)

        writer.write_table(table)

        batch.clear()

    if writer:
        writer.close()

    print(f"\nStep 01 Complete.")
    print(f"Dataset saved to: {output_path}")

    # ----------------------------
    # FINAL SUMMARY COMPUTATION
    # ----------------------------

    # Global
    df_global = pd.DataFrame(
        [{col: (global_sum[col] / global_count if global_count else 0) for col in numeric_cols}],
        index=["GLOBAL"],
    )

    # Language-level
    df_language = pd.DataFrame.from_dict(lang_sum, orient="index")

    for col in numeric_cols:
        df_language[col] = df_language[col] / df_language.index.map(lang_count)

    df_language.index.name = "language"

    # Granular (language / label)
    df_granular = pd.DataFrame.from_dict(granular_sum, orient="index")

    for col in numeric_cols:
        df_granular[col] = df_granular[col] / df_granular.index.map(granular_count)

    df_granular.index = pd.MultiIndex.from_tuples(df_granular.index, names=["language", "label"])

    # ----------------------------
    # WRITE REPORT (PRETTY VERSION)
    # ----------------------------

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== STEP 01 - METRICS EXTRACTION SUMMARY ===\n\n")
        f.write(f"Total Valid Unique Samples: {global_count}\n\n")

        f.write("--- Global Averages ---\n")
        f.write(df_global.to_string())
        f.write("\n\n")

        f.write("--- Language Averages ---\n")
        f.write(df_language.to_string())
        f.write("\n\n")

        f.write("--- Language / Label Averages ---\n")
        f.write(df_granular.to_string())

    print(f"Averages report saved to: {report_path}")
