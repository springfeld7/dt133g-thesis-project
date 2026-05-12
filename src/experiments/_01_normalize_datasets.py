"""_01_normalize_samples.py

Step 01: Normalization layer for robuster deduplication which removes comments.

Utilizes the TranStructIVER's CommentDeletionRule to strip comments while preserving code structure.

The VarCLR emebeddings are robust enough to handle whitespace and formatting variations, so we focus on
comment removal for normalization.

Since the TranStructIVER parser is used, samples that are considered unmeaningful are filtered out.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import pandas as pd
from tqdm import tqdm

from transtructiver.parsing.parser import Parser
from transtructiver.mutation.rules.comment_deletion import CommentDeletionRule
from transtructiver.mutation.mutation_context import MutationContext
from .utils.resource_manager import ResourceManager

# ----------------------------
# CONFIG
# ----------------------------

INPUT_DIR = Path("data/_00_extracted_datasets")
OUTPUT_DIR = Path("data/_01_normalized_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
MAX_WORKERS = ResourceManager.get_cpu_limit()

# ----------------------------
# WORKER INITIALIZATION
# ----------------------------


def init_worker():
    """
    Initializes heavy objects inside each process worker.

    This avoids pickling issues and ensures thread/process safety
    for TranStructIVER components.
    """
    global parser, comment_rule, context

    parser = Parser()
    comment_rule = CommentDeletionRule(level=3)
    context = MutationContext()


# ----------------------------
# HELPERS
# ----------------------------


def normalize_sample(code: str, lang: str) -> str | None:
    """
    Parses, removes comments, and returns normalized code.

    Args:
        code: Raw source code string to normalize.
        lang: Language of the source code.

    Returns:
        str | None: Normalized code string or None if parsing fails.
    """
    tree, _result = parser.parse(code, lang, annotate=False)
    if tree is None:
        return None

    for node in tree.traverse():
        if lang == "cpp" and node.type == "comment" and node.text:
            if node.text.startswith("//"):
                node.semantic_label = "line_comment"
            elif node.text.startswith("/*"):
                node.semantic_label = "block_comment"
        elif lang == "python":
            parent = node.parent
            if (
                node.type == "string"
                and parent
                and parent.type == "expression_statement"
                and parent.parent
                and parent.parent.type in {"module", "block"}
            ):
                if any(
                    child.type in {"string_start", "string_end"} and child.text in ('"""', "'''")
                    for child in node.children
                ):
                    node.semantic_label = "block_comment"
            elif node.type == "comment":
                node.semantic_label = "line_comment"
        elif lang == "java":
            if "comment" in node.type:
                node.semantic_label = node.type

    comment_rule.apply(tree, context)

    return tree.to_code()


def chunk_list(data, batch_size: int):
    """
    Splits a list into smaller batches.

    Args:
        data (list): Input list.
        batch_size (int): Size of each batch.

    Yields:
        list: Chunked batch of data.
    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def process_batch(batch):
    """
    Processes a batch of dataset rows in a single worker process.

    Each batch:
    - parses code
    - removes comments
    - filters invalid samples
    - returns normalized rows

    Args:
        batch (list): List of pandas row objects (itertuples).

    Returns:
        list[dict]: Cleaned and normalized rows.
    """
    results = []

    for row in batch:
        norm_code = normalize_sample(row["code"], row["language"])

        if norm_code is None or not str(norm_code).strip():
            continue

        results.append(
            {
                **row,
                "code_normalized": norm_code,
                "normalized_hash": hashlib.md5(norm_code.encode("utf-8")).hexdigest(),
            }
        )

    return results


# ----------------------------
# MAIN PIPELINE
# ----------------------------


def run_step_01():
    """
    Iterates through all parquets from Step 00, normalizes the code samples by removing comments,
    and saves the results to Step 01 directory.
    """
    files = list(INPUT_DIR.glob("*.parquet"))

    if not files:
        print(f"No parquet files found in {INPUT_DIR}")
        return

    for f in files:
        print(f"\nProcessing: {f.name}")
        df = pd.read_parquet(f)
        print(f"Initial samples: {len(df)}")

        rows = df.to_dict(orient="records")
        num_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
        batches = list(chunk_list(rows, BATCH_SIZE))

        valid_rows = []

        with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:

            for batch_result in tqdm(
                executor.map(process_batch, batches),
                total=num_batches,
                desc=f"Normalizing {f.stem}",
            ):
                valid_rows.extend(batch_result)

        df_clean = pd.DataFrame(valid_rows)

        print(f"Removed samples: {len(df) - len(df_clean)}")
        print(f"Remaining samples: {len(df_clean)}")

        df_clean.to_parquet(OUTPUT_DIR / f.name, index=False)


if __name__ == "__main__":
    run_step_01()
