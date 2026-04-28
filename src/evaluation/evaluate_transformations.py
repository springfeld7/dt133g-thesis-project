import json
import pandas as pd

from .CodeBLEU.calc_code_bleu import calc_code_bleu
from .varclr.models.encoders import Encoder


def get_varsim_score(manifest_path: str = "output/manifest.jsonl"):
    """Compute VarCLR similarity scores for rename mutations in a manifest."""
    model = Encoder.from_pretrained("varclr-codebert")

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            snippet = json.loads(line)
            entries = snippet.get("entries")

            originals = []
            renamed = []

            for entry in entries:
                history: list[dict[str, str]] = entry.get("history")
                for i in history:
                    if not i.get("action") == "RENAME":
                        continue
                    metadata = entry.get("metadata")
                    originals.append(metadata.get("old_val"))
                    renamed.append(metadata.get("new_val"))

            if originals and renamed:
                model.score(originals, renamed)


def get_code_bleu_score(parquet_path: str = "output/augmented_dataset.parquet"):
    """Compute CodeBLEU per language from the augmented dataset parquet file."""
    df: pd.DataFrame = pd.read_parquet(parquet_path)

    lang_original_df: pd.Series[list[str]] = df.groupby("language").apply(
        lambda x: x["original_code"].to_list()
    )
    lang_mutated_df: pd.Series[list[str]] = df.groupby("language").apply(
        lambda x: x["mutated_code"].to_list()
    )

    for lang in lang_original_df.keys():
        original_df: list[str] | None = lang_original_df.get(lang)
        mutated_df: list[str] | None = lang_mutated_df.get(lang)

        if original_df and mutated_df:
            for idx, code in enumerate(original_df):
                calc_code_bleu([code], mutated_df[idx], lang)
