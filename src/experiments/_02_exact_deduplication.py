"""_02_exact_deduplication.py

Step 02: Global Exact Deduplication.
Removes identical logic across all datasets using 'normalized_hash'.
Uses a balance-weighted selection to preserve the proportional integrity of smaller datasets.
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter

from .sample_selection.calculate_balance_score import calculate_balance_score

# ----------------------------
# CONFIG
# ----------------------------

INPUT_DIR = Path("data/_01_normalized_datasets")
OUTPUT_DIR = Path("data/_02_exact_deduplication")
REPORT_DIR = Path("output/")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# MAIN PIPELINE
# ----------------------------


def run_step_02():
    files = list(INPUT_DIR.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {INPUT_DIR}")
        return

    # LOAD DATA AND TRACK STATS
    print("Gathering dataset statistics...")
    initial_counts = {}
    all_dfs = {}
    for f in files:
        df = pd.read_parquet(f)
        initial_counts[f.stem] = len(df)
        all_dfs[f.stem] = df

    remaining_counts = initial_counts.copy()

    # GLOBAL REGISTRY
    registry = {}
    drop_indices = defaultdict(set)

    # Tracking for report
    collision_stats = defaultdict(lambda: {"lost_to": Counter(), "won_against": Counter()})

    # GLOBAL COMPETITION
    for ds_name, df in all_dfs.items():
        for row in tqdm(df.itertuples(), total=len(df), desc=f"Scanning {ds_name}"):
            h = row.normalized_hash
            idx = row.Index

            if h not in registry:
                registry[h] = {"ds": ds_name, "idx": idx}
            else:
                existing = registry[h]
                exist_ds = existing["ds"]

                curr_score = calculate_balance_score(ds_name, remaining_counts, initial_counts)
                exist_score = calculate_balance_score(exist_ds, remaining_counts, initial_counts)

                if curr_score > exist_score:
                    # Current DS is 'healthier', it drops the row
                    drop_indices[ds_name].add(idx)
                    remaining_counts[ds_name] -= 1
                    collision_stats[ds_name]["lost_to"][exist_ds] += 1
                    collision_stats[exist_ds]["won_against"][ds_name] += 1
                else:
                    # Existing DS in registry is healthier, it drops the row
                    drop_indices[exist_ds].add(existing["idx"])
                    remaining_counts[exist_ds] -= 1
                    collision_stats[exist_ds]["lost_to"][ds_name] += 1
                    collision_stats[ds_name]["won_against"][exist_ds] += 1

                    # Update registry: current DS now 'owns' this hash
                    registry[h] = {"ds": ds_name, "idx": idx}

    # SAVE AND SUMMARIZE
    final_counts = {}
    print("\nApplying drops and saving...")
    for ds_name, df in all_dfs.items():
        indices_to_drop = list(drop_indices[ds_name])
        df_cleaned = df.drop(index=indices_to_drop).reset_index(drop=True)

        out_path = OUTPUT_DIR / f"{ds_name}.parquet"
        df_cleaned.to_parquet(out_path, index=False)
        final_counts[ds_name] = len(df_cleaned)

    # WRITE SUMMARY REPORT
    print("\nWriting summary report...")
    write_summary_report(initial_counts, final_counts, collision_stats)
    print("Step 02 complete.")


def write_summary_report(initial, final, collisions):
    report_path = REPORT_DIR / "_02_exact_deduplication_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== GLOBAL EXACT DEDUPLICATION REPORT ===\n\n")

        global_start = sum(initial.values())
        global_end = sum(final.values())

        f.write(f"Global Totals: {global_start} -> {global_end}\n")
        f.write(
            f"Total Unique Logic Samples: {len(initial) if global_start == 0 else global_end}\n"
        )
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

            # Show who this dataset "competed" with
            lost_total = sum(collisions[ds]["lost_to"].values())
            if lost_total > 0:
                f.write(f"  Top losses to:\n")
                for other, count in collisions[ds]["lost_to"].most_common(3):
                    f.write(f"    - {other}: {count}\n")
            f.write("\n")

    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    run_step_02()
