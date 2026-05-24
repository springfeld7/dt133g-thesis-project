# Reproducing Project Results

This repository contains the TranStructIVer pipeline and supporting code used in the experiments. This README explains how to reproduce the main results: preparing data, running the pipeline, fine-tuning/evaluation, and locating outputs.

**Prerequisites**
- **Python**: 3.10 & 3.14+.
- **Repository**: clone or open this repo at the project root.
- **Source on PYTHONPATH**: many convenience commands use `src` as the import root.

**Environment Setup**
- **Create virtual environment**: create and activate a venv (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- **Install code in editable mode** (installs dependencies from `pyproject.toml`):

```bash
pip install -e .
```

**Quick Repro (end-to-end)**
- Notes:
  - Dataset preprocessing scripts live in [src/preprocessing](src/preprocessing).
  - Key preprocessing steps used in the experiments: [src/preprocessing/_00_extract_datasets.py](src/preprocessing/_00_extract_datasets.py), [src/preprocessing/_01_normalize_datasets.py](src/preprocessing/_01_normalize_datasets.py), and [src/preprocessing/_06_generate_splits.py](src/preprocessing/_06_generate_splits.py).
  - Use notebook [notebooks/run_finetune](notebooks/run_finetune.ipynb) for fine-tuning.
  - Use notebook [notebooks/run_experiments](notebooks/run_experiments.ipynb) for full experiment pipeline.

**TranStructIVer**
- For CLI usage, options, and outputs, see [src/transtructiver/README.md](src/transtructiver/README.md).

**Fine-tuning & Experiments (quick steps)**
- Config files for fine-tuning live in `configs/fine_tune/` (for example, [configs/fine_tune/codebert-base.json](configs/fine_tune/codebert-base.json)).
- Notebooks are in the `notebooks/` folder. Typical interactive workflow:

  1. Open `notebooks/run_finetune.ipynb` in Jupyter Lab or VS Code.
  2. Inspect and set the input variables at the top of the notebook (dataset paths, output dir, config file).
  3. Run cells in order (use `Run All` or step through cell-by-cell) to execute preprocessing, training, and evaluation blocks.

**Evaluation**
- Evaluation scripts and CodeBLEU tools are under [src/evaluation](src/evaluation).

- CodeBLEU helper scripts live in [src/evaluation/CodeBLEU](src/evaluation/CodeBLEU).

**Outputs and Where to Find Results**
- Pipeline outputs are written into `data/` and `output/` by default.
- The repository already includes example outputs and aggregated results in the `output/` folder (reports and figures).

**Repro Tips & Troubleshooting**
- Ensure `PYTHONPATH=src` (or install package editable) so CLI modules import correctly.
- Check `transtructiver.config.yaml` and config files in `configs/` to reproduce exact hyperparameters and rule settings.

**Contacts & References**
- Pipeline implementation: [src/transtructiver](src/transtructiver)
- Preprocessing scripts: [src/preprocessing](src/preprocessing)
- Experiments / notebooks: [notebooks](notebooks)
