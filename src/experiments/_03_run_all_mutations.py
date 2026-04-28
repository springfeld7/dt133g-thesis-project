"""_03_run_all_mutations.py

Step three of experiments: Run all mutation transformations on the final samples.

Each transformation is executed independently across predefined parameter
levels to measure sensitivity of downstream models.
"""

from pathlib import Path
import subprocess

# -----------------------------
# CONFIGURATION
# -----------------------------

INPUT_FILE = Path("data/final_samples/final_samples.parquet")
BASE_OUTPUT_DIR = Path("output/step03_mutations")

PYTHON_CMD = "python"

# -----------------------------
# EXPERIMENT MATRIX
# -----------------------------
# Each entry = one independent run

EXPERIMENTS = [
    # -------------------------
    # rename-identifier
    # -------------------------
    {
        "name": "rename_l0",
        "rules": ["rename-identifier"],
        "params": {"rename-identifier": {"level": 0}},
    },
    {
        "name": "rename_l1",
        "rules": ["rename-identifier"],
        "params": {"rename-identifier": {"level": 1}},
    },
    {
        "name": "rename_l2",
        "rules": ["rename-identifier"],
        "params": {"rename-identifier": {"level": 2}},
    },
    {
        "name": "rename_l3",
        "rules": ["rename-identifier"],
        "params": {"rename-identifier": {"level": 3}},
    },

    # -------------------------
    # comment deletion
    # -------------------------
    {
        "name": "comment_deletion",
        "rules": ["comment-deletion"],
    },

    # -------------------------
    # comment normalization
    # -------------------------
    {
        "name": "comment_normalization_l0",
        "rules": ["comment-normalization"],
        "params": {"comment-normalization": {"level": 0}},
    },

    # -------------------------
    # whitespace normalization
    # -------------------------
    {
        "name": "whitespace_l0",
        "rules": ["whitespace-normalization"],
        "params": {"whitespace-normalization": {"level": 0}},
    },
    {
        "name": "whitespace_l1",
        "rules": ["whitespace-normalization"],
        "params": {"whitespace-normalization": {"level": 1}},
    },

    # -------------------------
    # dead-code-insertion
    # -------------------------
    {
        "name": "dead_code_l0",
        "rules": ["dead-code-insertion"],
        "params": {"dead-code-insertion": {"level": 0}},
    },
    {
        "name": "dead_code_l1",
        "rules": ["dead-code-insertion"],
        "params": {"dead-code-insertion": {"level": 1}},
    },
    {
        "name": "dead_code_l2",
        "rules": ["dead-code-insertion"],
        "params": {"dead-code-insertion": {"level": 2}},
    },
    {
        "name": "dead_code_l3",
        "rules": ["dead-code-insertion"],
        "params": {"dead-code-insertion": {"level": 3}},
    },

    # -------------------------------
    # control structure substitution
    # -------------------------------
    {
        "name": "control_structure_substitution",
        "rules": ["control-structure-substitution"],
    },
]

# -----------------------------
# EXECUTION
# -----------------------------

def run_experiment(exp: dict):
    """Run a single mutation experiment via CLI pipeline."""

    output_dir = BASE_OUTPUT_DIR / exp["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "cli",
        str(INPUT_FILE),
        *exp["rules"],
        "--output-dir",
        str(output_dir),
    ]

    # add rule parameters if present
    params = exp.get("params", {})
    for rule, rule_params in params.items():
        for key, value in rule_params.items():
            cmd += ["--rule-param", f"{rule}:{key}={value}"]

    print("\n==============================")
    print(f"Running experiment: {exp['name']}")
    print(" ".join(cmd))
    print("==============================\n")

    subprocess.run(cmd, check=True)


def main():
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== STEP 03: MUTATION EXPERIMENTS ===\n")
    print(f"Input dataset: {INPUT_FILE}")
    print(f"Output base:   {BASE_OUTPUT_DIR}")
    print(f"Total runs:    {len(EXPERIMENTS)}\n")

    for exp in EXPERIMENTS:
        run_experiment(exp)

    print("\n=== STEP 03 COMPLETE ===")


if __name__ == "__main__":
    main()
