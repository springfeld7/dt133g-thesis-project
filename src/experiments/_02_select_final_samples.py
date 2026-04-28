"""_02_select_final_samples.py

Step two of experiments: Select final evaluation samples from validated dataset.

This script constructs a 10,200-sample evaluation dataset using:
- hard validity constraints (structural feasibility)
- distribution-aware trimming (per-language 95th percentile)
- stratified sampling across language and label

The pipeline ensures:
- sufficient structural complexity for transformations
- removal of extreme-length artifacts
- balanced representation across languages and labels
- reproducible sampling via fixed seed
"""

from pathlib import Path
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------

INPUT_PATH = Path("data/valid_samples/valid_samples.parquet")
OUTPUT_PATH = Path("data/final_samples/final_samples.parquet")

RANDOM_SEED = 42

TARGET_TOTAL = 5100
LANGUAGES = ["cpp", "java", "python"]
LANGUAGE_MAP = {
    "C++": "cpp",
    "Java": "java",
    "Python": "python",
}

N_LANG = TARGET_TOTAL // len(LANGUAGES)
N_PER_GROUP = N_LANG // 2

# -----------------------------
# HARD CONSTRAINTS
# -----------------------------

MIN_LLOC = 50
MIN_CHAR = 1000
MAX_CHAR_ABSOLUTE = 6000


# -----------------------------
# THRESHOLD COMPUTATION
# -----------------------------


def compute_percentile_bounds(df: pd.DataFrame) -> dict:
    """
    Compute language-specific upper bounds using 95th percentile
    after applying hard constraints.

    Args:
        df (pd.DataFrame): DataFrame containing valid samples.

    Returns:
        dict: Dictionary with language as key and bounds as value.
    """

    bounds = {}

    for lang in LANGUAGES:
        lang_df = df[df["language"] == lang]

        bounds[lang] = {
            "char_max": min(lang_df["char_count"].quantile(0.95), MAX_CHAR_ABSOLUTE),
            "lloc_max": lang_df["lloc"].quantile(0.95),
        }

    return bounds


# -----------------------------
# PIPELINE
# -----------------------------


def run_step_02():
    """
    Step 02: Final sample selection pipeline.
    """

    print("\n--- Step 02: Final Sample Selection ---")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_parquet(INPUT_PATH)

    df["language"] = df["language"].map(LANGUAGE_MAP)

    print("\n=== INITIAL DATASET ===")
    print(f"Total samples: {len(df)}")

    # -----------------------------
    # HARD VALIDITY FILTERING
    # -----------------------------
    df = df[
        (df["lloc"] >= MIN_LLOC)
        & (df["char_count"] >= MIN_CHAR)
        & (df["char_count"] <= MAX_CHAR_ABSOLUTE)
    ].copy()

    print("\n=== AFTER HARD FILTERING ===")
    print(f"Remaining samples: {len(df)}")

    # -----------------------------
    # PERCENTILE-BASED BOUNDS
    # -----------------------------
    bounds = compute_percentile_bounds(df)

    print("\n=== PERCENTILE BOUNDS ===")
    for lang, b in bounds.items():
        print(f"{lang}: {b}")

    # -----------------------------
    # APPLY DISTRIBUTION TRIMMING
    # -----------------------------
    filtered_parts = []

    for lang in LANGUAGES:
        lang_df = df[df["language"] == lang]
        b = bounds[lang]

        lang_df = lang_df[
            (lang_df["lloc"] <= b["lloc_max"]) & (lang_df["char_count"] <= b["char_max"])
        ]

        filtered_parts.append(lang_df)

    df = pd.concat(filtered_parts).reset_index(drop=True)

    print("\n=== AFTER PERCENTILE TRIMMING ===")
    print(f"Remaining samples: {len(df)}")

    # -----------------------------
    # STRATIFIED SAMPLING
    # -----------------------------
    sampled_parts = []

    for lang in LANGUAGES:
        lang_df = df[df["language"] == lang]

        for label in ["HUMAN_GENERATED", "MACHINE_GENERATED"]:
            subset = lang_df[lang_df["label"] == label]

            if len(subset) < N_PER_GROUP:
                raise ValueError(
                    f"Insufficient samples for {lang} / {label}: "
                    f"required={N_PER_GROUP}, available={len(subset)}"
                )

            sampled_parts.append(subset.sample(n=N_PER_GROUP, random_state=RANDOM_SEED))

    final_df = pd.concat(sampled_parts).reset_index(drop=True)

    # -----------------------------
    # 5. FINAL CHECKS
    # -----------------------------
    print("\n=== FINAL DATASET ===")
    print(f"Total samples: {len(final_df)}")

    print("\n=== DISTRIBUTION ===")
    print(final_df.groupby(["language", "label"]).size())

    print("\n=== LLOC STATS ===")
    print(final_df.groupby(["language", "label"])["lloc"].describe())

    print("\n=== CHAR COUNT STATS ===")
    print(final_df.groupby(["language", "label"])["char_count"].describe())

    # -----------------------------
    # SAVE
    # -----------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUTPUT_PATH, index=False)

    print("\nSaved dataset to:", OUTPUT_PATH)


if __name__ == "__main__":
    run_step_02()
