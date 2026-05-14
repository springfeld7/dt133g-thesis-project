"""_08_map_gradient.py

Step 08 of experiments: Cumulatively apply transformation tiers.

"""

from operator import indexOf
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
# Each entry = one independent run

EXPERIMENTS = [
    # -----------------------------------
    # Tier 1 — Surface-level normalization
    # -----------------------------------
    {
        "name": "tier_1",
        "rules": [
            "whitespace-normalization",
            "rename-identifier",
            "comment-normalization",
        ],
        "params": {
            "whitespace-normalization": {"level": 0},
            "rename-identifier": {"level": 0},
            "comment-normalization": {"level": 0},
        },
    },
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
    print("-------------------------------\n")
    print(f"Running experiment: {exp["name"]}")

    files = list(INPUT_DIR.glob("*/test.parquet"))

    if not files:
        print("No files found.")
        return

    for file in files:
        print(f"\nProcessing: {file.parent.name}/{file.name}")

        base_dir = BASE_OUTPUT_DIR / file.parent.name
        output_dir = base_dir / f"{exp["name"]}"
        output_dir.mkdir(parents=True, exist_ok=True)

        this_tier_file = output_dir / "augmented_dataset.parquet"
        if this_tier_file.exists():
            print(
                f"\nAugmentet dataset for {file.parent.name} already exists at:\n{this_tier_file}"
            )
            print("\n-------------------------------")
            return

        _INPUT = file

        exp_idx = indexOf(EXPERIMENTS, exp)
        if exp_idx >= 0:
            # get saved file with previous tier mutations to avoid reapplying them
            prev_tier_file = base_dir / f"tier_{exp_idx}" / "augmented_dataset.parquet"
            if prev_tier_file.exists():
                _INPUT = prev_tier_file
                print(f"\n  Fetched previous tier dataset from {prev_tier_file}\n")

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
        print("\n-------------------------------\n")

        subprocess.run(cmd, check=True)


def main():
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== STEP 08: MAPPING GRADIENT ===\n")

    for exp in EXPERIMENTS:
        run_experiment(exp)

    print("\n=== STEP 08 COMPLETE ===")


if __name__ == "__main__":
    main()
