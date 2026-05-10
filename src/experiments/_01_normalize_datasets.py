"""_01_normalize_samples.py

Step 01: Normalization layer for robuster deduplication which removes comments.

Utilizes the TranStructIVER's CommentDeletionRule to strip comments while preserving code structure.

The VarCLR emebeddings are robust enough to handle whitespace and formatting variations, so we focus on
comment removal for normalization.

Since the TranStructIVER parser is used, samples that are considered unmeaningful are filtered out.
"""

from pathlib import Path
import hashlib
import pandas as pd
from tqdm import tqdm

from transtructiver.parsing.parser import Parser
from transtructiver.mutation.rules.comment_deletion import CommentDeletionRule
from transtructiver.mutation.mutation_context import MutationContext

# ----------------------------
# CONFIG
# ----------------------------

INPUT_DIR = Path("data/_00_extract_datasets")
OUTPUT_DIR = Path("data/_01_normalized_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# HELPERS
# ----------------------------


def normalize_sample(
    parser: Parser,
    comment_rule: CommentDeletionRule,
    code: str,
    lang: str,
    context: MutationContext,
) -> str | None:
    """
    Parses, removes comments, and returns normalized code.

    Args:
        parser: TranStructIVER parser instance.
        comment_rule: Instance of CommentDeletionRule for comment removal.
        code: Raw source code string to normalize.
        lang: Language of the source code.
        context: Mutation context for handling transformations.
    Returns:
        str | None: Normalized code string or None if parsing fails.
    """
    tree, _result = parser.parse(code, lang)

    if tree is None:
        return None

    comment_rule.apply(tree, context)

    return tree.to_code()


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

    parser = Parser()
    context = MutationContext()
    comment_rule = CommentDeletionRule(level=3)

    for f in files:
        print(f"\nProcessing: {f.name}")
        df = pd.read_parquet(f)
        print(f"Initial samples: {len(df)}")

        valid_rows = []
        removed_samples = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Normalizing {f.stem}"):

            norm_code = normalize_sample(
                parser, comment_rule, row["code"], row.get("language"), context
            )

            # Remove invalid / unmeaningful samples
            if norm_code is None or str(norm_code).strip() == "":
                removed_samples += 1
                continue

            row = row.copy()

            row["code_normalized"] = norm_code
            row["normalized_hash"] = hashlib.md5(norm_code.encode("utf-8")).hexdigest()

            valid_rows.append(row)

        df_clean = pd.DataFrame(valid_rows)

        print(f"Removed {removed_samples} invalid samples from {f.stem}")
        print(f"Remaining samples: {len(df_clean)}")

        df_clean.to_parquet(OUTPUT_DIR / f.name, index=False)


if __name__ == "__main__":
    run_step_01()
