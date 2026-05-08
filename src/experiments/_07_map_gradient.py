"""_07_map_gradient.py

Step 07 of experiments: Cumulatively apply transformation tiers.

"""

from datetime import datetime
from operator import indexOf
from pathlib import Path
import subprocess


# -----------------------------
# CONFIGURATION
# -----------------------------

INPUT_FILE = Path("data/final_samples/final_samples.parquet")
BASE_OUTPUT_DIR = Path("output/transformations")

PYTHON_CMD = "python"

# -----------------------------
# EXPERIMENT MATRIX
# -----------------------------
# Each entry = one independent run

EXPERIMENTS = [
    # -----------------------------------
    # Tier 1 — Surface-level normalization
    # -----------------------------------
    # {
    #     "name": "tier_1",
    #     "rules": [
    #         "whitespace-normalization",
    #         "rename-identifier",
    #         "comment-normalization",
    #     ],
    #     "params": {
    #         "whitespace-normalization": {"level": 0},
    #         "rename-identifier": {"level": 0},
    #         "comment-normalization": {"level": 0},
    #     },
    # },
    # -----------------------------------
    # Tier 2 — Lexical drift (Tier 1 + Tier 2)
    # -----------------------------------
    {
        "name": "tier_2",
        "rules": [
            "rename-identifier",
            "comment-normalization",
        ],
        "params": {
            "rename-identifier": {"level": 1},
            "comment-normalization": {"level": 1},
        },
    },
    # -----------------------------------
    # Tier 3 — Structural rewrites (Tier 1 + 2 + 3)
    # -----------------------------------
    {
        "name": "tier_3",
        "rules": [
            "rename-identifier",
            "dead-code-insertion",
            "control-structure-substitution",
        ],
        "params": {
            "rename-identifier": {"level": 2},
            "dead-code-insertion": {"level": 0},
        },
    },
    # -----------------------------------
    # Tier 4 — Heavy obfuscation (Tier 1 + 2 + 3 + 4)
    # -----------------------------------
    {
        "name": "tier_4",
        "rules": [
            "whitespace-normalization",
            "rename-identifier",
            "dead-code-insertion",
            "comment-deletion",
        ],
        "params": {
            "whitespace-normalization": {"level": 1},
            "rename-identifier": {"level": 3},
            "dead-code-insertion": {"level": 1},
        },
    },
]

# -----------------------------
# EXECUTION
# -----------------------------


def run_experiment(exp: dict):
    """Run a tiered experiment via CLI pipeline."""
    print("\n==============================")
    print(f"Running experiment: {exp['name']}")

    output_dir = BASE_OUTPUT_DIR / exp["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    _INPUT = INPUT_FILE

    exp_idx = indexOf(EXPERIMENTS, exp)
    if exp_idx >= 0:
        # get saved file with previous tier mutations to avoid reapplying them
        prev_tier_file = BASE_OUTPUT_DIR / f"tier_{exp_idx+1}" / "augmented_dataset.parquet"
        if prev_tier_file.exists():
            _INPUT = prev_tier_file
            print(f"  Fetched previous tier dataset from {prev_tier_file}")

    cmd = [
        "uv",
        "run",
        "cli",
        str(_INPUT),
        *exp["rules"],
        "--output-dir",
        str(output_dir),
    ]

    # add rule parameters if present
    params = exp.get("params", {})
    for rule, rule_params in params.items():
        for key, value in rule_params.items():
            cmd += ["--rule-param", f"{rule}:{key}={value}"]

    print(" ".join(cmd))
    print("==============================\n")

    subprocess.run(cmd, check=True)


def main():
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== STEP 06: MAPPING GRADIENT ===\n")
    print(f"Input dataset: {INPUT_FILE}")
    print(f"Output base:   {BASE_OUTPUT_DIR}")
    print(f"Total runs:    {len(EXPERIMENTS)}\n")

    for exp in EXPERIMENTS:
        run_experiment(exp)

    print("\n=== STEP 04 COMPLETE ===")


if __name__ == "__main__":
    main()
