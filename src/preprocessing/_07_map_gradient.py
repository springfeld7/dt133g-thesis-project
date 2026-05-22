"""_07_map_gradient.py

Step 07 of experiments: Cumulatively apply transformation tiers.

"""

from .utils import env_init
from .utils.resource_manager import ResourceManager
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
            "whitespace-normalization",
            "rename-identifier",
            "comment-normalization",
        ],
        "params": {
            "whitespace-normalization": {"level": 0},
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
            "whitespace-normalization",
            "rename-identifier",
            "comment-normalization",
            "dead-code-insertion",
            "control-structure-substitution",
        ],
        "params": {
            "whitespace-normalization": {"level": 0},
            "rename-identifier": {"level": 2},
            "comment-normalization": {"level": 1},
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
            "comment-normalization",
            "dead-code-insertion",
            "control-structure-substitution",
            "comment-deletion",
        ],
        "params": {
            "whitespace-normalization": {"level": 1},
            "rename-identifier": {"level": 3},
            "comment-normalization": {"level": 1},
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

        this_tier_file = output_dir / "augmented_dataset.parquet"
        if this_tier_file.exists():
            print(
                f"\nAugmentet dataset for {file.parent.name} already exists at:\n{this_tier_file}"
            )
            print("\n-------------------------------")
            return

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
                cmd += ["--rule-param", f"{rule}:{key}={value}", "--workers", f"{ResourceManager.get_cpu_limit()//2}"]

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
