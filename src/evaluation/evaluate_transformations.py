import json

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
