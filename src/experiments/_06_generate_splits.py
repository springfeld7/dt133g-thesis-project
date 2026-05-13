"""_06_generate_splits.py

Step 06: Train/Test Split Generation Layer

This stage:
1. Loads filtered datasets from Step 05
2. Generates reproducible train/test splits
3. Builds a strictly controlled evaluation (test) set:
    - 10,000 samples per dataset
    - balanced 50/50 label distribution
    - ERROR == 0 only
    - stratified across language + label
4. Train set contains remaining samples (no balancing enforced)
5. Writes split datasets + audit report
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------

INPUT_DIR = Path("data/_05_filtered_datasets")
OUTPUT_DIR = Path("data/_06_generated_splits")
REPORT_PATH = Path("output/_06_generated_split_reports.txt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 10_000
RANDOM_SEED = 42

# ----------------------------
# HELPERS
# ----------------------------


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Stratified sampling ensuring:
    - Exact total size n
    - 50/50 label balance
    - Approx. uniform language distribution per label
    - Deterministic output

    Args:
        df: Input dataframe (must contain 'language', 'label')
        n: Total samples (must be even)
        seed: Random seed

    Returns:
        Stratified sample of size n
    """

    if n % 2 != 0:
        raise ValueError("n must be even for 50/50 label balance")

    rng = np.random.default_rng(seed)

    # Split target evenly across labels (strict 50/50 constraint)
    per_label = n // 2

    out = []

    for label in [0, 1]:
        # Filter current label group
        label_df = df[df["label"] == label]

        # Group by language inside this label
        groups = list(label_df.groupby("language"))

        # Base quota per language (ensures near-uniform distribution)
        base = per_label // len(groups)

        # Remaining samples distributed one-by-one
        remainder = per_label % len(groups)

        # Shuffle group order so remainder is not biased
        order = rng.permutation(len(groups))

        parts = []

        for i, idx in enumerate(order):
            lang, sub = groups[idx]

            # Assign base quota + remainder if applicable
            target = base + (1 if i < remainder else 0)

            # Safety check: prevent sampling more than available
            if len(sub) < target:
                raise ValueError(f"Insufficient samples for {lang} / {label}")

            # Deterministic sampling within group
            parts.append(sub.sample(n=target, random_state=seed))

        # Combine all languages for this label
        out.append(pd.concat(parts))

    # Combine both labels and shuffle final dataset
    result = pd.concat(out)

    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def write_report(f, name, df, train_df, test_df) -> None:
    """
    Writes one dataset section into an already open report file.

    Args:
        f: Open file handle for the report (write mode).
        name: Dataset name used as section header.
        df: Original full dataset.
        train_df: Train split DataFrame.
        test_df: Test split DataFrame.
    """

    f.write(f"\n\n=== DATASET: {name} ===\n\n")

    f.write("=== TOTAL ===\n")
    f.write(f"Samples: {len(df)}\n\n")

    f.write("=== TEST SET ===\n")
    f.write(f"Samples: {len(test_df)}\n")
    f.write(f"ERROR = 0 only: {test_df['ERROR'].sum() == 0}\n\n")

    f.write("Label distribution:\n")
    for k, v in test_df["label"].value_counts().items():
        f.write(f"  {k}: {v}\n")

    f.write("\nLanguage distribution:\n")
    for k, v in test_df["language"].value_counts().items():
        f.write(f"  {k}: {v}\n")

    f.write("\nLanguage-Label distribution:\n")
    for (lang, label), count in test_df.groupby(["language", "label"]).size().items():
        f.write(f"  {lang} | {label}: {count}\n")

    f.write("\n=== TRAIN SET ===\n")
    f.write(f"Samples: {len(train_df)}\n\n")

    f.write("Label distribution:\n")
    for k, v in train_df["label"].value_counts().items():
        f.write(f"  {k}: {v}\n")

    f.write("\nLanguage distribution:\n")
    for k, v in train_df["language"].value_counts().items():
        f.write(f"  {k}: {v}\n")

    f.write("\n----------------------------\n")


# ----------------------------
# MAIN PIPELINE
# ----------------------------


def run_step_06():
    """
    Generates train/test splits from Step 05 outputs.
    """

    files = list(INPUT_DIR.glob("*.parquet"))

    if not files:
        print("No files found.")
        return

    print("\n=== STEP 06: SPLIT GENERATION ===")

    report_data = []

    for file in files:
        print(f"\nProcessing: {file.name}")

        df = pd.read_parquet(file)

        # ----------------------------
        # TEST SET (STRICT RULES)
        # ----------------------------

        test_pool = df[df["ERROR"] == 0].copy()

        if len(test_pool) < TEST_SIZE:
            raise ValueError(
                f"Not enough ERROR=0 samples in {file.name}: "
                f"{len(test_pool)} available, {TEST_SIZE} required"
            )

        test_df = stratified_sample(test_pool, TEST_SIZE)

        # ----------------------------
        # TRAIN SET (SUBSET WILL BE USED AS VALIDATION IN FINE-TUNING)
        # ----------------------------

        train_df = df.drop(test_df.index).reset_index(drop=True)

        # ----------------------------
        # SAVE
        # ----------------------------

        out_dir = OUTPUT_DIR / file.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_parquet(out_dir / "train.parquet", index=False)
        test_df.to_parquet(out_dir / "test.parquet", index=False)

        report_data.append((file.stem, df, train_df, test_df))

    # ----------------------------
    # REPORT (single pass)
    # ----------------------------

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== STEP 06 - TRAIN/TEST SPLIT REPORT ===\n")

        for name, df, train_df, test_df in report_data:
            write_report(f, name, df, train_df, test_df)

    print("\nStep 06 complete.")


# ----------------------------
# ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    run_step_06()
