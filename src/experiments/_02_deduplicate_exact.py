"""_02_deduplicate_exact.py

Step 02: Deduplication inter- and intra-dataset.
Uses a balance-weighted selection to preserve the proportional integrity of smaller datasets.
Checks both raw code hashes and normalized logic hashes.
"""

import hashlib
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter

from .utils.calculate_balance_score import calculate_balance_score

# ----------------------------
# CONFIG
# ----------------------------

INPUT_DIR = Path("data/_01_normalized_datasets")
OUTPUT_DIR = Path("data/_02_exact_deduplicated_datasets")
REPORT_PATH = Path("output/_02_exact_deduplication_report.txt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------
# HELPER
# ----------------------------


def _generate_hash(code: str) -> str:
    """
    Generates a hash for the raw, original code.

    Args:
        code (str): The source code string to hash.

    Returns:
        str: A hexadecimal hash string representing the input code.
    """
    return hashlib.md5(code.encode("utf-8")).hexdigest()


# ----------------------------
# MAIN PIPELINE
# ----------------------------


def run_step_02():
    """
    Executes the deduplication process across all datasets in the input directory.
    """
    files = list(INPUT_DIR.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {INPUT_DIR}")
        return

    # LOAD DATA AND GENERATE HASHES
    print("Loading datasets and generating hashes...")
    initial_counts = {}
    all_dfs = {}

    for f in files:
        df = pd.read_parquet(f)

        df["hash"] = df["code"].apply(_generate_hash)
        df["hash_normalized"] = df["code_normalized"].apply(_generate_hash)

        initial_counts[f.stem] = len(df)
        all_dfs[f.stem] = df

    hash_counts = Counter()
    for df in all_dfs.values():
        hash_counts.update(df["hash_normalized"].astype(str).tolist())

    remaining_counts = initial_counts.copy()
    registry_raw = {}
    registry_norm = {}
    drop_indices = defaultdict(set)
    pair_records = []
    pair_id = 0

    # Tracking for report
    collision_stats = defaultdict(lambda: {"lost_to": Counter(), "won_against": Counter()})

    # GLOBAL COMPETITION
    for ds_name, df in all_dfs.items():
        for row in tqdm(df.itertuples(), total=len(df), desc=f"Scanning {ds_name}"):
            h_raw = row.hash
            h_norm = row.hash_normalized
            idx = row.Index

            # Check for any collision
            existing = registry_raw.get(h_raw) or registry_norm.get(h_norm)

            if not existing:
                # Register new unique entry
                registry_raw[h_raw] = {"ds": ds_name, "idx": idx}
                registry_norm[h_norm] = {"ds": ds_name, "idx": idx}
            else:
                # COLLISION DETECTED
                exist_ds = existing["ds"]
                curr_score = calculate_balance_score(ds_name, remaining_counts, initial_counts)
                exist_score = calculate_balance_score(exist_ds, remaining_counts, initial_counts)

                if curr_score > exist_score:
                    # Current DS is healthier, it drops the row
                    drop_indices[ds_name].add(idx)
                    remaining_counts[ds_name] -= 1
                    collision_stats[ds_name]["lost_to"][exist_ds] += 1
                    pair_records.append(
                        {
                            "pair_id": pair_id,
                            "pair_type": "exact",
                            "group_id": h_norm,
                            "group_size": hash_counts[str(h_norm)],
                            "normalized_hash": h_norm,
                            "removed_dataset": ds_name,
                            "removed_idx": idx,
                            "removed_row_index": idx,
                            "removed_sample_id": f"{ds_name}::{idx}",
                            "kept_dataset": exist_ds,
                            "kept_idx": existing["idx"],
                            "kept_row_index": existing["idx"],
                            "survivor_sample_id": f"{exist_ds}::{existing['idx']}",
                            "removed_score": curr_score,
                            "kept_score": exist_score,
                            "similarity": 1.0,
                            "winner": exist_ds,
                        }
                    )
                    pair_id += 1
                else:
                    # Existing DS in registry drops the row
                    drop_indices[exist_ds].add(existing["idx"])
                    remaining_counts[exist_ds] -= 1
                    collision_stats[exist_ds]["lost_to"][ds_name] += 1
                    pair_records.append(
                        {
                            "pair_id": pair_id,
                            "pair_type": "exact",
                            "group_id": h_norm,
                            "group_size": hash_counts[str(h_norm)],
                            "normalized_hash": h_norm,
                            "removed_dataset": exist_ds,
                            "removed_idx": existing["idx"],
                            "removed_row_index": existing["idx"],
                            "removed_sample_id": f"{exist_ds}::{existing['idx']}",
                            "kept_dataset": ds_name,
                            "kept_idx": idx,
                            "kept_row_index": idx,
                            "survivor_sample_id": f"{ds_name}::{idx}",
                            "removed_score": exist_score,
                            "kept_score": curr_score,
                            "similarity": 1.0,
                            "winner": ds_name,
                        }
                    )
                    pair_id += 1

                    # Current DS takes ownership of these hashes
                    entry = {"ds": ds_name, "idx": idx}
                    registry_raw[h_raw] = entry
                    registry_norm[h_norm] = entry

    # SAVE AND SUMMARIZE
    final_counts = {}
    cleaned_dfs = {}
    print("\nApplying drops, cleaning columns, and saving...")
    for ds_name, df in all_dfs.items():
        indices_to_drop = list(drop_indices[ds_name])
        df_cleaned = df.drop(index=indices_to_drop).reset_index(drop=True)

        out_path = OUTPUT_DIR / f"{ds_name}.parquet"
        df_cleaned.to_parquet(out_path, index=False)
        final_counts[ds_name] = len(df_cleaned)
        cleaned_dfs[ds_name] = df_cleaned

    pair_df = pd.DataFrame(pair_records)
    pair_df.to_parquet(
        OUTPUT_DIR / "_02_dup_pairs" / "_02_exact_duplicate_pairs.parquet", index=False
    )
    pair_df.to_csv(OUTPUT_DIR / "_02_dup_pairs" / "_02_exact_duplicate_pairs.csv", index=False)

    # WRITE SUMMARY REPORT
    print("\nWriting summary report...")
    write_summary_report(initial_counts, final_counts, collision_stats, cleaned_dfs)
    print("Step 02 complete.")


def write_summary_report(initial, final, collisions, datasets):

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== GLOBAL DEDUPLICATION REPORT ===\n\n")

        global_start = sum(initial.values())
        global_end = sum(final.values())

        f.write(f"Global Totals: {global_start} -> {global_end}\n")
        f.write(f"Total Unique Logic Samples: {global_end}\n")
        f.write(f"Global Reduction: {100*(1 - global_end/global_start):.2f}%\n\n")

        f.write("--- PER-DATASET BREAKDOWN ---\n")

        for ds in sorted(initial.keys()):
            start = initial[ds]
            end = final[ds]
            removed = start - end

            f.write(f"{ds.upper()}:\n")
            f.write(f"  Initial: {start}\n")
            f.write(f"  Final:   {end}\n")
            f.write(f"  Removed: {removed} ({100*(removed/start) if start > 0 else 0:.2f}%)\n")

            # Collision statistics
            lost_total = sum(collisions[ds]["lost_to"].values())
            if lost_total > 0:
                f.write("  Top losses to:\n")

                for other, count in collisions[ds]["lost_to"].most_common(3):
                    f.write(f"    - {other}: {count}\n")

            # Final distributions
            df = datasets[ds]

            lang_dist = dict(Counter(df["language"]))
            label_dist = dict(Counter(df["label"]))
            lang_label_dist = defaultdict(Counter)

            for lang, label in zip(df["language"], df["label"]):
                lang_label_dist[lang][label] += 1

            f.write("\n  Language distribution:\n")
            for lang, count in lang_dist.items():
                f.write(f"    {lang}: {count}\n")

            f.write("\n  Label distribution:\n")
            for label, count in label_dist.items():
                f.write(f"    {label}: {count}\n")

            f.write("\n  Language-label distribution:\n")
            for lang, label_counter in lang_label_dist.items():
                human = label_counter.get(0, 0)
                ai = label_counter.get(1, 0)

                f.write(f"    {lang}: " f"human={human}, " f"ai={ai}\n\n")

    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    run_step_02()
