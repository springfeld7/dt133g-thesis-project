"""_00_extract_datasets.py

Step 0: Dataset extraction for multi-source code classification experiments.

This module standardizes heterogeneous Hugging Face code datasets into a unified schema:
    - code: raw source code string
    - language: normalized programming language (python, java, cpp)
    - label: binary classification label (0 = human, 1 = AI-generated)

The script also produces dataset-level statistics and writes a global extraction report
to support reproducibility and dataset quality analysis.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor

from .utils.dataset_manager import DatasetManager
from .utils.resource_manager import ResourceManager


# ----------------------------
# CONFIGURATION
# ----------------------------

DATASETS = {
    "droidcollection": "DaniilOr/DroidCollection",
    "codet_m4": "DaniilOr/CoDET-M4",
}

OUTPUT_DIR = Path("data/_00_extracted_datasets")
REPORT_PATH = Path("output/_00_extracted_datasets_report.txt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 10_000


# ----------------------------
# NORMALIZATION FUNCTIONS
# ----------------------------


def _normalize_language(lang: str) -> str | None:
    """
    Normalizes raw language labels into a unified set.

    Supported outputs:
        - python
        - java
        - cpp

    Args:
        lang (str): Raw language label from dataset.

    Returns:
        str | None: Normalized language string or None if unsupported.
    """
    if not isinstance(lang, str):
        return None

    lang = lang.strip().lower()

    if lang in ["python"]:
        return "python"
    if lang in ["java"]:
        return "java"
    if lang in ["c++", "cpp"]:
        return "cpp"

    return None


def _normalize_label(value, dataset: str) -> int | None:
    """
    Normalizes dataset-specific label formats into binary labels.

    Mapping:
        - 0 → human-written code
        - 1 → AI-generated code

    Args:
        value: Raw label value from dataset entry.
        dataset (str): Dataset name for schema-specific interpretation.

    Returns:
        int | None: Normalized label or None if invalid/unrecognized.
    """
    if dataset == "droidcollection":
        if value == "HUMAN_GENERATED":
            return 0
        if value == "MACHINE_GENERATED":
            return 1

    if dataset == "codet_m4":
        v = str(value).lower()
        if "human" in v:
            return 0
        if "ai" in v:
            return 1

    if dataset == "ai_detector":
        if value in [0, 1]:
            return int(value)

    return None


def _extract_code(entry: dict) -> str | None:
    """
    Extracts source code from a dataset entry.

    Handles multiple possible field names across datasets.

    Args:
        entry (dict): Dataset sample.

    Returns:
        str | None: Code string or None if missing.
    """
    return entry.get("code") or entry.get("Code")


def _extract_language(entry: dict) -> str | None:
    """
    Extracts programming language field from dataset entry.

    Args:
        entry (dict): Dataset sample.

    Returns:
        str | None: Raw language string or None if missing.
    """
    return entry.get("language") or entry.get("Language") or entry.get("language_name")


def _extract_label(entry: dict) -> str | None:
    """
    Extracts label field from dataset entry.

    Args:
        entry (dict): Dataset sample.

    Returns:
        str | None: Raw label value (type varies per dataset), or None if missing.
    """
    if "label" in entry:
        label = entry["label"]
    elif "Label" in entry:
        label = entry["Label"]
    elif "target" in entry:
        label = entry["target"]
    else:
        label = None

    return label


# ----------------------------
# WORKER FUNCTION
# ----------------------------


def process_dataset(item) -> dict:
    """
    Worker wrapper for parallel dataset processing.

    Each process must create its own DatasetManager instance
    to avoid shared state / authentication / streaming issues.

    Args:
        item (tuple): (name, repo_id) for the dataset to process.

    Returns:
        dict: Summary statistics and distribution information for the dataset.
    """
    name, repo_id = item

    manager = DatasetManager()
    manager.authenticate()

    print(f"\n--- Processing {name} ---")

    manager.set_repo(repo_id)
    stream = manager.get_stream(split="train")

    output_path = OUTPUT_DIR / f"{name}.parquet"

    batch = []
    writer = None
    processed_count = 0

    stats = {
        "total": 0,
        "kept": 0,
        "dropped_lang": 0,
        "dropped_label": 0,
    }

    lang_counter = defaultdict(int)
    label_counter = defaultdict(int)
    label_lang_counter = defaultdict(lambda: defaultdict(int))

    for entry in stream:
        stats["total"] += 1

        code = _extract_code(entry)
        lang = _extract_language(entry)
        label = _extract_label(entry)

        if not isinstance(code, str) or str(code).strip() == "":
            continue

        norm_lang = _normalize_language(lang)
        if norm_lang is None:
            stats["dropped_lang"] += 1
            continue

        norm_label = _normalize_label(label, name)
        if norm_label is None:
            stats["dropped_label"] += 1
            continue

        batch.append(
            {
                "code": code,
                "language": norm_lang,
                "label": norm_label,
            }
        )

        stats["kept"] += 1
        lang_counter[norm_lang] += 1
        label_counter[norm_label] += 1
        label_lang_counter[norm_lang][norm_label] += 1

        if len(batch) >= BATCH_SIZE:
            table = pa.Table.from_pandas(pd.DataFrame(batch))

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)
            processed_count += len(batch)
            print(f"[{name}] Written {processed_count:,} samples...")
            batch.clear()

    # Final flush for the last partial batch
    if batch:
        table = pa.Table.from_pandas(pd.DataFrame(batch))
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

    print(f"--- Finished: {name} ---")

    return {
        "name": name,
        "stats": stats,
        "lang": dict(lang_counter),
        "label": dict(label_counter),
        "label_lang": {lang: dict(labels) for lang, labels in label_lang_counter.items()},
        "output_path": str(output_path),
    }


# ----------------------------
# REPORT GENERATION
# ----------------------------


def write_report(results: list[dict]):
    """
    Writes a global dataset extraction report.

    The report includes:
        - Per-dataset sample counts
        - Filtering statistics (dropped vs kept)
        - Language distribution
        - Label distribution
        - Global aggregation summary

    Args:
        results (list[dict]): List of dataset processing results.
    """
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== DATASET EXTRACTION REPORT ===\n\n")

        global_total = 0
        global_kept = 0

        for r in results:
            s = r["stats"]

            global_total += s["total"]
            global_kept += s["kept"]

            f.write(f"--- {r['name']} ---\n")
            f.write(f"Total samples: {s['total']}\n")
            f.write(f"Kept samples: {s['kept']}\n")
            f.write(f"Dropped (language): {s['dropped_lang']}\n")
            f.write(f"Dropped (label): {s['dropped_label']}\n")

            f.write("Language distribution:\n")
            for k, v in r["lang"].items():
                f.write(f"  {k}: {v}\n")

            f.write("\nLabel distribution:\n")
            for k, v in r["label"].items():
                f.write(f"  {k}: {v}\n")

            f.write("\nLabel distribution per language:\n")
            for lang, labels in r["label_lang"].items():
                human = labels.get(0, 0)
                ai = labels.get(1, 0)
                f.write(f"  {lang}: " f"human={human}, " f"ai={ai}\n")

            f.write("\n----------------------------\n\n")

        f.write("=== GLOBAL SUMMARY ===\n")
        f.write(f"Total processed: {global_total}\n")
        f.write(f"Total kept: {global_kept}\n")

    print(f"\nReport written to: {REPORT_PATH}")


# ----------------------------
# ENTRY POINT
# ----------------------------


def run_step_00():
    """
    Executes dataset extraction for each specified dataset in parallel and generates a report.
    """
    items = list(DATASETS.items())
    results = []

    max_workers = min(len(items), ResourceManager.get_cpu_limit())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_dataset, items):
            results.append(result)

    write_report(results)


if __name__ == "__main__":
    run_step_00()
