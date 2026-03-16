import importlib
import json

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "transtructiver.reporting.output_manager",
        "transtructiver.prototype.reporting.output_manager",
    ],
)
def test_runstats_success_rate(module_name):
    module = importlib.import_module(module_name)

    stats = module.RunStats(parsed_ok=3, parse_skipped=1, verified_ok=2, verified_fail=2)

    assert stats.processed == 4
    assert stats.success_rate == 0.5


@pytest.mark.parametrize(
    "module_name",
    [
        "transtructiver.reporting.output_manager",
        "transtructiver.prototype.reporting.output_manager",
    ],
)
def test_output_paths_summary_single_file(module_name, tmp_path):
    module = importlib.import_module(module_name)

    manager = module.OutputManager(str(tmp_path), max_rows_per_shard=0, compress_output=False)
    manifest_path, dataset_path, summary_path = manager.output_paths_summary()

    assert manifest_path.endswith("manifest.jsonl")
    assert dataset_path.endswith("augmented_dataset.parquet")
    assert summary_path.endswith("summary_log.csv")


@pytest.mark.parametrize(
    "module_name",
    [
        "transtructiver.reporting.output_manager",
        "transtructiver.prototype.reporting.output_manager",
    ],
)
def test_output_manager_writes_manifest_and_dataset(module_name, tmp_path):
    module = importlib.import_module(module_name)

    with module.OutputManager(
        str(tmp_path), max_rows_per_shard=0, compress_output=False
    ) as manager:
        manager.write_manifest(0, "row_0", [{"node_id": [0, 0], "history": [], "metadata": {}}])
        manager.write_dataset_row(0, "row_0", "original", "mutated")

    manifest_file = tmp_path / "manifest.jsonl"
    dataset_file = tmp_path / "augmented_dataset.parquet"
    summary_file = tmp_path / "summary_log.csv"

    assert manifest_file.exists()
    assert dataset_file.exists()
    assert summary_file.exists()

    payload = json.loads(manifest_file.read_text(encoding="utf-8").strip())
    assert payload["snippet_id"] == "row_0"
    assert payload["entries"][0]["node_id"] == [0, 0]
