"""_04_extract_stats.py

Step 04: Extraction of statistical features and error annotation for valid/invalid parsing.

This stage:
1. Loads deduplicated datasets from Step 03
2. Adds ERROR column based on parse validity
3. Adds statistical features (char_count, loc, lloc, identifier_density, for_loop_density, comment_density, whitespace_ratio)
4. Computes statistical summaries:
   - Global
   - Per label
   - Per language
   - Per (language, label)
5. Saves datasets + summary report
"""

from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from .utils.analysis import SampleAnalyzer
from .utils.resource_manager import ResourceManager


# ----------------------------
# CONFIG
# ----------------------------

INPUT_DIR = Path("data/_03_near_deduplicated_datasets")
OUTPUT_DIR = Path("data/_04_extracted_stats_datasets")
REPORT_PATH = Path("output/_04_extracted_stats_report.txt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = ResourceManager.get_cpu_limit()


# ----------------------------
# STATS HELPERS
# ----------------------------

NUMERIC_COLS = [
    "char_count",
    "loc",
    "lloc",
    "identifier_density",
    "for_loop_density",
    "comment_density",
    "whitespace_ratio",
]

# ----------------------------
# WORKER STATE
# ----------------------------


def init_worker():
    """
    Initializes analyzer inside each worker process.
    """
    global analyzer
    analyzer = SampleAnalyzer()


def process_sample(args):
    """
    Processes a single sample in parallel.

    Args:
        args (tuple): (code, lang, label)

    Returns:
        dict: extracted metrics
    """
    code, lang, label = args

    tree = analyzer.get_valid_tree(code, lang, label)
    metrics = analyzer.calculate_metrics(code, tree)

    return metrics


# ----------------------------
# MAIN PIPELINE
# ----------------------------


def run_step_04():
    """
    Extract statistics and extend datasets with ERROR + metric annotations.

    Args:
        None
    Returns:
        None
    """
    files = list(INPUT_DIR.glob("*.parquet"))

    if not files:
        print("No files found.")
        return

    REPORT_PATH.write_text("", encoding="utf-8")

    for file in files:
        print(f"\nProcessing: {file.name}")
        df = pd.read_parquet(file).reset_index(drop=True)

        codes = df["code"].to_numpy()
        langs = df["language"].to_numpy()
        labels = df["label"].to_numpy()

        inputs = list(zip(codes, langs, labels))

        raw_chunks = max(1, len(df) // (max(1, MAX_WORKERS) * 16))
        chunksize = max(1, min(1024, raw_chunks))
        print(f"Using chunksize={chunksize} for executor.map (raw={raw_chunks})")
        sys.stdout.flush()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:

            metrics_list = list(
                tqdm(
                    executor.map(process_sample, inputs, chunksize=chunksize),
                    total=len(df),
                    desc="Analyzing",
                )
            )

        # Integration
        df_metrics = pd.DataFrame.from_records(metrics_list)
        df = df.join(df_metrics)

        # Aggregation
        error_count = int(df["ERROR"].sum())
        error_ratio = df["ERROR"].mean()
        global_avg = df[NUMERIC_COLS].mean().to_dict()
        label_avg = df.groupby("label")[NUMERIC_COLS].mean().to_dict("index")
        lang_avg = df.groupby("language")[NUMERIC_COLS].mean().to_dict("index")
        pair_avg = df.groupby(["language", "label"])[NUMERIC_COLS].mean().to_dict("index")
        label_error_ratio = df.groupby("label")["ERROR"].mean().to_dict()

        # Save and report
        out_path = OUTPUT_DIR / file.name
        df.to_parquet(out_path, index=False)
        _write_report(
            file=file,
            df=df,
            global_avg=global_avg,
            label_avg=label_avg,
            lang_avg=lang_avg,
            pair_avg=pair_avg,
            error_count=error_count,
            error_ratio=error_ratio,
            label_error_ratio=label_error_ratio,
        )


def _write_report(
    file, df, global_avg, label_avg, lang_avg, pair_avg, error_count, error_ratio, label_error_ratio
):
    """
    Handles the text formatting for the statistical report.

    Args:
        file (Path): The original file path.
        df (pd.DataFrame): The processed dataframe.
        global_avg (dict): Global metric averages.
        label_avg (dict): Averages grouped by label.
        lang_avg (dict): Averages grouped by language.
        pair_avg (dict): Averages grouped by (language, label).
        error_count (int): Total count of ERROR=1.
        error_ratio (float): Mean of ERROR column.
        label_error_ratio (dict): ERROR ratio per label.
    Returns:
        None
    """
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("=== STEP 04 - DATASET STATISTICS ===\n\n")
        f.write(f"Dataset: {file.name}\n")
        f.write(f"Total samples: {len(df)}\n")

        f.write(f"GLOBAL ERROR samples: {error_count}\n")
        f.write(f"GLOBAL ERROR ratio: {error_ratio:.4f}\n\n")

        f.write("--- ERROR RATIO PER LABEL ---\n")

        for label, ratio in label_error_ratio.items():
            f.write(f"{label}: {ratio:.4f}\n")

        f.write("\n" + "=" * 40 + "\n")
        f.write("--- METRIC AVERAGES ---\n")

        f.write("--- GLOBAL AVERAGES ---\n")
        for k, v in global_avg.items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\n--- PER LABEL AVERAGES ---\n")
        for label, vals in label_avg.items():
            f.write(f"\n[{label}]\n")
            for k, v in vals.items():
                f.write(f"{k}: {v:.6f}\n")

        f.write("\n--- PER LANGUAGE AVERAGES ---\n")
        for lang, vals in lang_avg.items():
            f.write(f"\n[{lang}]\n")
            for k, v in vals.items():
                f.write(f"{k}: {v:.6f}\n")

        f.write("\n--- PER (LANGUAGE, LABEL) AVERAGES ---\n")
        for (lang, label), vals in pair_avg.items():
            f.write(f"\n[{lang} | {label}]\n")
            for k, v in vals.items():
                f.write(f"{k}: {v:.6f}\n\n")


if __name__ == "__main__":
    run_step_04()
