import json
import re
import pandas as pd
from tqdm import tqdm

from ..transtructiver.mutation.rules.comment_deletion import CommentDeletionRule
from transtructiver.parsing.parser import Parser
from transtructiver.mutation.mutation_engine import MutationEngine

from .CodeBLEU.calc_code_bleu import calc_code_bleu
from .CodeBLEU.parser.utils import remove_comments_and_docstrings
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


def collapse_to_one_line(code: str, lang: str) -> str:
    if lang == "python":
        code = code.replace("\n", ";")
    else:
        code = code.replace("\n", " ")
    return code


def strip_cpp_imports(code):
    cleaned = []
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith("#include"):
            continue
        if s.startswith("#define"):
            continue
        if s.startswith("#ifdef") or s.startswith("#ifndef") or s.startswith("#endif"):
            continue
        if s.startswith("using namespace"):
            continue
        if s.startswith("using std::"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def get_code_bleu_score(parquet_path):
    """Compute CodeBLEU per language from the augmented dataset parquet file."""
    df = pd.read_parquet(parquet_path)
    tier = parquet_path.parent.name
    dataset_name = parquet_path.parent.parent.name

    results = []
    no_data_flow = []

    for row in tqdm(
        df.itertuples(),
        total=len(df),
        desc=f"Scoring snippets [{dataset_name.replace('_', '-').upper()} - {tier.replace('_', ' ').title()}]",
    ):
        try:
            ref_lines = remove_comments_and_docstrings(str(row.original_code), str(row.language))
            hyp_lines = remove_comments_and_docstrings(str(row.mutated_code), str(row.language))
        except Exception as e:
            no_data_flow.append(row)
            ref_lines = str(row.original_code)
            hyp_lines = str(row.mutated_code)

        if row.language == "cpp":
            ref_lines = strip_cpp_imports(ref_lines)
            hyp_lines = strip_cpp_imports(hyp_lines)

        ref_one_line = collapse_to_one_line(ref_lines, str(row.language))
        hyp_one_line = collapse_to_one_line(hyp_lines, str(row.language))

        scores = calc_code_bleu(ref_one_line, hyp_one_line, str(row.language))

        if scores.get("dataflow_match_score") == 0:
            no_data_flow.append(row)

        res = {
            "dataset": dataset_name,
            "tier": tier,
            "snippet_id": str(row.snippet_id),
            "language": str(row.language),
        }

        for k, v in scores.items():
            res[k] = str(v)

        results.append(res)

    return results, no_data_flow
