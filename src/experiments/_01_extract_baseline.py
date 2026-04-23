"""_01_extract_baseline.py

Step one of experiments: Extract the samples used for the baseline.
"""

import hashlib
import os
from pathlib import Path
import pandas as pd
from .sample_selection.dataset_manager import DatasetManager
from .sample_selection.analysis import SampleAnalyzer


def run_step_01():
    """
    Step 01: Initial Filtering, Metric Extraction, and Persistent Storage.

    1. Streams data and filters for valid parse trees.
    2. Extracts stylistic metrics.
    3. Deduplicates via MD5 hashing.
    4. Saves the pool to root/data/valid_samples/filtered_pool.parquet.
    5. Saves a summary report to root/output/dataset_summary.txt.
    """
    manager = DatasetManager()
    analyzer = SampleAnalyzer()

    manager.authenticate()
    stream = manager.get_iterator()

    features = stream.features
    print(f"\n[Dataset Shape/Schema]")
    print(f"Columns: {list(features.keys())}")

    total_rows = stream.info.splits["train"].num_examples
    print(f"Total Rows (Metadata): {total_rows}")

    filtered_data = []
    print("--- Step 01: Streaming & Initial Filtering ---")

    for entry in stream:

        lang = entry.get("Language")
        label = entry.get("Label")
        code = entry.get("Code", "")
        valid_tree = analyzer.get_valid_tree(code, lang, label)

        if valid_tree:
            # Metric Extraction
            row_data = analyzer.calculate_metrics(code, lang, label, valid_tree)

            row_data["code"] = code
            row_data["code_hash"] = hashlib.md5(code.encode("utf-8")).hexdigest()
            row_data["language"] = lang
            row_data["label"] = label

            filtered_data.append(row_data)

            if len(filtered_data) % 1000 == 0:
                print(f"Collected {len(filtered_data)} valid candidates...")

    df = pd.DataFrame(filtered_data)

    # Remove Duplicates by code_hash
    df = df.drop_duplicates(subset="code_hash").reset_index(drop=True)
    # Assign clean Index (0 to N-1)
    df.insert(0, "index", range(len(df)))

    # Organize Columns
    columns_order = [
        "index",
        "code",
        "language",
        "label",
        "char_count",
        "loc",
        "lloc",
        "identifier_density",
        "for_loop_density",
        "comment_density",
        "whitespace_ratio",
        "code_hash",
    ]
    df = df[columns_order]

    # --- IO OPERATIONS ---
    data_path = Path("data/valid_samples/filtered_pool.parquet")
    output_path = Path("output/dataset_summary.txt")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the Parquet file to data/valid_samples
    df.to_parquet(data_path, index=False)

    # Calculate Averages for the report
    numeric_cols = [
        "char_count",
        "loc",
        "lloc",
        "identifier_density",
        "for_loop_density",
        "comment_density",
        "whitespace_ratio",
    ]

    # Define the grouping levels for the report
    stats_granular = df.groupby(["language", "label"])[numeric_cols].mean()
    stats_language = df.groupby(["language"])[numeric_cols].mean()
    stats_global = df[numeric_cols].mean().to_frame().T
    stats_global.index = ["GLOBAL"]

    # Write the summary report to output/dataset_summary.txt
    with open(output_path, "w") as f:
        f.write("===  STEP 01 - METRICS EXTRACTION SUMMARY ===\n")
        f.write(f"Initial Samples Processed: {total_rows}\n")
        f.write(f"Total Valid Unique Samples: {len(df)}\n\n")

        f.write("--- Averages per Language/Label ---\n")
        f.write(stats_granular.to_string())
        f.write("\n\n")

        f.write("--- Averages per Language (Combined Labels) ---\n")
        f.write(stats_language.to_string())
        f.write("\n\n")

        f.write("--- Global Averages (Combined Everything) ---\n")
        f.write(stats_global.to_string())

    print(f"\nStep 01 Complete.")
    print(f"Pool saved to: {data_path}")
    print(f"Averages report saved to: {output_path}")


if __name__ == "__main__":
    run_step_01()
