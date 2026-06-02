"""_08_ablation_single_tier.py

Step 08 of experiments: Transform a single tier using a 'leave‑one‑out' ablation approach.

"""

from ..preprocessing.utils import env_init
from ..preprocessing.utils.resource_manager import ResourceManager
from pathlib import Path
import subprocess


# -----------------------------
# CONFIGURATION
# -----------------------------

INPUT_DIR = Path("data/_06_generated_splits")
BASE_OUTPUT_DIR = Path("data/transformations")

PYTHON_CMD = "python"

# -----------------------------
# EXPERIMENT MATRIX
# -----------------------------

EXPERIMENTS = [
    # -----------------------------------
    # Tier 1 — Whitespace Normalization Removed
    # -----------------------------------
    {
        "name": "tier_1_no_ws_norm",
        "rules": [
            "rename-identifier",
            "comment-normalization",
        ],
        "params": {
            "rename-identifier": {"level": 0},
            "comment-normalization": {"level": 0},
        },
    },
    # -----------------------------------
    # Tier 1 — Rename identifier Removed
    # -----------------------------------
    {
        "name": "tier_1_no_rename",
        "rules": [
            "whitespace-normalization",
            "comment-normalization",
        ],
        "params": {
            "whitespace-normalization": {"level": 0},
            "comment-normalization": {"level": 0},
        },
    },
    # -----------------------------------
    # Tier 1 — Comment Normalization Removed
    # -----------------------------------
    {
        "name": "tier_1_no_comment_norm",
        "rules": [
            "whitespace-normalization",
            "rename-identifier",
        ],
        "params": {
            "whitespace-normalization": {"level": 0},
            "rename-identifier": {"level": 0},
        },
    },
]

# -----------------------------
# EXECUTION
# -----------------------------


def run_experiment(exp: dict):
    """Run a tiered experiment via CLI pipeline."""
    print("-------------------------------\n")
    print(f"Running experiment: {exp['name']}")

    files = list(INPUT_DIR.glob("*/test.parquet"))

    if not files:
        print("No files found.")
        return

    for file in files:
        print(f"\nProcessing: {file.parent.name}/{file.name}")

        base_dir = BASE_OUTPUT_DIR / file.parent.name
        output_dir = base_dir / f"{exp['name']}"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "cli",
            str(file),
            *exp["rules"],
            "--output-dir",
            str(output_dir),
        ]

        # add rule parameters if present
        params = exp.get("params", {})
        for rule, rule_params in params.items():
            for key, value in rule_params.items():
                cmd += [
                    "--rule-param",
                    f"{rule}:{key}={value}",
                    "--workers",
                    f"{ResourceManager.get_cpu_limit()//2}",
                ]

        print(" ".join(cmd))
        print("\n-------------------------------\n")

        subprocess.run(cmd, check=True)


def main():
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== STEP 08: PERFORMING TRANSFORMATIONS ===\n")

    for exp in EXPERIMENTS:
        run_experiment(exp)

    print("\n=== STEP 08 COMPLETE ===")


if __name__ == "__main__":
    main()
