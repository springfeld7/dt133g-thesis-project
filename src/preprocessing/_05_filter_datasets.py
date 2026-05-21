"""_05_filter_datasets.py

Step 05: Final Dataset Curation Layer

This stage:
1. Loads datasets from Step 04
2. Applies quality filtering using LLOC constraints
3. Applies per-language label balancing (human vs AI)
4. Produces the final curated datasets used for split generation
5. Writes detailed audit reports for reproducibility

Filtering constraints:
    MIN_LLOC = 6
    MAX_LLOC = 300

Balancing strategy:
    - Per-language balancing
    - Keeps equal number of human and AI samples
    - Preserves language composition
    - Random but reproducible downsampling
"""

from pathlib import Path
from typing import TextIO
from collections import Counter, defaultdict
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------

INPUT_DIR = Path("data/_04_extracted_stats_datasets")
OUTPUT_DIR = Path("data/_05_filtered_datasets")
REPORT_PATH = Path("output/_05_filtered_datasets_report.txt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

MIN_LLOC = 6
MAX_LLOC = 300

RANDOM_SEED = 42


# ----------------------------
# HELPERS
# ----------------------------


def build_lang_label_stats(data: pd.DataFrame):
    """
    Builds distribution statistics for a dataset subset.

    Args:
        data (pd.DataFrame):
            Dataset subset.

    Returns:
        tuple:
            (
                language_counter,
                label_counter,
                language_label_counter
            )
    """
    lang = Counter(data["language"])
    label = Counter(data["label"])

    pair = defaultdict(Counter)

    for l, y in zip(data["language"], data["label"]):
        pair[l][y] += 1

    return lang, label, pair


def write_distribution_block(
    f: TextIO, title: str, lang: Counter, label: Counter, pair: defaultdict
) -> None:
    """
    Writes a formatted distribution block to the report.

    Args:
        f (TextIO):
            Open report file handle.

        title (str):
            Section title.

        lang (Counter):
            Language distribution.

        label (Counter):
            Label distribution.

        pair (defaultdict):
            Language-label distribution.
    """
    f.write(f"\n=== {title} ===\n")

    f.write("Language:\n")
    for k, v in sorted(lang.items()):
        f.write(f"  {k}: {v}\n")

    f.write("\nLabel:\n")
    for k, v in sorted(label.items()):
        f.write(f"  {k}: {v}\n")

    f.write("\nLanguage-Label:\n")
    for language, dist in sorted(pair.items()):
        f.write(f"  {language}: " f"human={dist.get(0, 0)}, " f"ai={dist.get(1, 0)}\n")


# ----------------------------
# MAIN
# ----------------------------


def run_step_05():
    """
    Executes final dataset curation.

    Pipeline:
        1. Apply LLOC filtering
        2. Apply per-language label balancing
        3. Save curated datasets
        4. Generate audit report
    """
    files = list(INPUT_DIR.glob("*.parquet"))

    if not files:
        print("No files found.")
        return

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== STEP 05 FILTERED DATASETS REPORT ===\n\n")

        for file in files:
            print(f"Processing: {file.name}")

            df = pd.read_parquet(file).copy()

            # Preserve original identity BEFORE any filtering
            df["_orig_idx"] = df.index

            # ----------------------------
            # LLOC FILTERING
            # ----------------------------

            mask = (df["lloc"] >= MIN_LLOC) & (df["lloc"] <= MAX_LLOC)

            # Split dataset into:
            # - df_after_filter: valid complexity range
            # - df_filtered: rejected by LLOC constraints
            df_after_filter = df[mask].copy()
            df_filtered = df[~mask].copy()

            # ----------------------------
            # PER-LANGUAGE BALANCING
            # ----------------------------

            balanced_parts = []

            # We balance within each language independently
            for language, sub in df_after_filter.groupby("language"):

                # Split into binary classes
                human = sub[sub["label"] == 0]
                ai = sub[sub["label"] == 1]

                # Maximum possible balanced size is limited by minority class
                n = min(len(human), len(ai))

                # Randomly sample equal amounts from both classes
                balanced_parts.append(human.sample(n=n, random_state=RANDOM_SEED))
                balanced_parts.append(ai.sample(n=n, random_state=RANDOM_SEED))

            # Combine all balanced language groups into final dataset
            df_kept = pd.concat(balanced_parts, ignore_index=False)

            # Shuffle rows for training stability (does NOT affect distribution)
            df_kept = df_kept.sample(frac=1, random_state=RANDOM_SEED)

            # Identify which original rows survived balancing
            kept_indices = set(df_kept["_orig_idx"])

            # Everything in df_after_filter not in kept_indices
            # is considered removed due to balancing step
            df_balance_removed = df_after_filter[
                ~df_after_filter["_orig_idx"].isin(kept_indices)
            ].copy()

            # Cleanup helper column before saving
            df_kept = df_kept.drop(columns=["_orig_idx"]).reset_index(drop=True)
            df_balance_removed = df_balance_removed.drop(columns=["_orig_idx"]).reset_index(
                drop=True
            )

            # ----------------------------
            # SAVE
            # ----------------------------

            out_path = OUTPUT_DIR / file.name
            df_kept.to_parquet(out_path, index=False)

            # ----------------------------
            # BUILD REPORT STATS
            # ----------------------------

            # Full original dataset distributions
            lang_all, label_all, pair_all = build_lang_label_stats(df)

            # Final curated dataset distributions
            lang_kept, label_kept, pair_kept = build_lang_label_stats(df_kept)

            # Only LLOC filtering effect
            lang_filtered, label_filtered, pair_filtered = build_lang_label_stats(df_filtered)

            # Only balancing removal effect
            lang_bal, label_bal, pair_bal = build_lang_label_stats(df_balance_removed)

            # ----------------------------
            # REPORT
            # ----------------------------

            f.write(f"--- {file.stem} ---\n")

            f.write(f"Total samples: {len(df)}\n")
            f.write(f"After LLOC filtering: {len(df_after_filter)}\n")
            f.write(f"Final kept samples: {len(df_kept)}\n")

            f.write(f"Filtered by LLOC constraints: {len(df_filtered)}\n")
            f.write(f"Removed by balancing: {len(df_balance_removed)}\n")

            # ALL DATA
            write_distribution_block(
                f,
                title="ALL",
                lang=lang_all,
                label=label_all,
                pair=pair_all,
            )

            # AFTER LLOC FILTER
            write_distribution_block(
                f,
                title="FILTERED (LLOC)",
                lang=lang_filtered,
                label=label_filtered,
                pair=pair_filtered,
            )

            # REMOVED BY BALANCING
            write_distribution_block(
                f,
                title="BALANCE REMOVED",
                lang=lang_bal,
                label=label_bal,
                pair=pair_bal,
            )

            # FINAL OUTPUT
            write_distribution_block(
                f,
                title="FINAL KEPT",
                lang=lang_kept,
                label=label_kept,
                pair=pair_kept,
            )

            f.write("\n----------------------------\n\n")

    print(f"Report written to {REPORT_PATH}")


# ----------------------------
# ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    run_step_05()
