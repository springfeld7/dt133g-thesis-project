"""Tests for identifier frequency map generation."""

from collections import Counter

from transtructiver.mutation.rules.utils.identifier_frequency_map import (
    BuildStats,
    IdentifierCounterMap,
    RoleTypeMap,
    _parse_args,
    build_frequency_payload,
    build_identifier_frequency_map,
)
from transtructiver.node import Node


def _wire(node: Node, parent: Node | None = None) -> Node:
    node.parent = parent
    for child in node.children:
        _wire(child, node)
    return node


def test_build_frequency_payload_with_fixed_top_n():
    type_map = {
        "number": Counter({"idx": 10, "count": 7, "tmp": 3}),
        "string": Counter({"arg": 4, "value": 2}),
    }
    role_type_map: RoleTypeMap = {"variable_name": type_map}
    stats = BuildStats(rows_seen=10, rows_parsed=8, rows_skipped=2, identifiers_counted=26)

    payload = build_frequency_payload(
        {"python": role_type_map},
        stats,
        source_parquet="samples.parquet",
        top_n=2,
        min_coverage=0.95,
    )

    assert payload["selection"]["mode"] == "fixed-top-n"
    assert payload["role_maps"]["python"]["variable_name"]["number"] == {"idx": 10, "count": 7}
    assert payload["role_maps"]["python"]["variable_name"]["string"] == {"arg": 4, "value": 2}


def test_build_frequency_payload_with_auto_coverage():
    type_map: IdentifierCounterMap = {
        "number": Counter({"one": 6, "two": 2, "three": 1, "four": 1}),
    }
    role_type_map: RoleTypeMap = {"variable_name": type_map}
    stats = BuildStats(rows_seen=4, rows_parsed=4, rows_skipped=0, identifiers_counted=10)

    payload = build_frequency_payload(
        {"python": role_type_map},
        stats,
        source_parquet="samples.parquet",
        top_n=None,
        min_coverage=0.8,
    )

    # 6 + 2 reaches 80% cumulative coverage in this distribution.
    assert payload["selection"]["mode"] == "coverage-auto"
    assert payload["role_maps"]["python"]["variable_name"]["number"] == {"one": 6, "two": 2}
    assert payload["selection"]["min_coverage"] == 0.8


def test_build_identifier_frequency_map_integration(monkeypatch, tmp_path):
    class DummyLoader:
        def iter_snippets(self, batch_size, start_index):
            assert batch_size == 100
            assert start_index == 0
            return [
                (0, "x = 1", "python"),
                (1, "def f(p): return p", "python"),
            ]

    class DummyParser:
        def parse(self, code, _):
            if code.startswith("x"):
                root = Node(
                    (0, 0),
                    (0, 0),
                    "module",
                    children=[Node((0, 0), (0, 1), "identifier", text="variable")],
                )
                root.semantic_label = "root"
                root.language = "python"
                _wire(root)
                root.children[0].semantic_label = "variable_name"
                root.children[0].context_type = "number"
                return root, None

            root = Node(
                (0, 0),
                (0, 0),
                "module",
                children=[
                    Node((0, 0), (0, 1), "identifier", text="function"),
                    Node((0, 0), (0, 1), "identifier", text="parameter"),
                    Node((0, 0), (0, 1), "identifier", text="parameter"),
                ],
            )
            root.semantic_label = "root"
            root.language = "python"
            _wire(root)
            root.children[0].semantic_label = "function_name"
            root.children[0].context_type = "none"
            root.children[1].semantic_label = "parameter_name"
            root.children[1].context_type = "none"
            root.children[2].semantic_label = "parameter_name"
            root.children[2].context_type = "none"
            return root, None

    import transtructiver.mutation.rules.utils.identifier_frequency_map as module

    monkeypatch.setattr(module, "DataLoader", lambda *args, **kwargs: DummyLoader())
    monkeypatch.setattr(module, "Parser", lambda: DummyParser())

    out = tmp_path / "identifier_map.json"
    payload = build_identifier_frequency_map(
        parquet_path="dummy.parquet",
        output_json_path=str(out),
        batch_size=100,
        top_n=1,
    )

    assert out.exists()
    assert payload["role_maps"]["python"]["variable_name"]["number"] == {"variable": 1}
    assert payload["role_maps"]["python"]["parameter_name"]["none"] == {"parameter": 2}
    assert payload["role_maps"]["python"]["function_name"]["none"] == {"function": 1}


def test_parse_args_loads_all_values_from_config(tmp_path):
    config_path = tmp_path / "identifier-map.json"
    config_path.write_text(
        """
{
    "identifier_frequency_map": {
        "parquet_path": "dataset.parquet",
        "output_json_path": "out/map.json",
        "batch_size": 250,
        "top_n": 12,
        "min_coverage": 0.87
    }
}
""".strip(),
        encoding="utf-8",
    )

    args = _parse_args(["--config", str(config_path)])

    assert args.parquet_path == "dataset.parquet"
    assert args.output_json_path == "out/map.json"
    assert args.batch_size == 250
    assert args.top_n == 12
    assert args.min_coverage == 0.87


def test_parse_args_cli_overrides_config_values(tmp_path):
    config_path = tmp_path / "identifier-map.json"
    config_path.write_text(
        """
{
    "identifier_frequency_map": {
        "parquet_path": "dataset.parquet",
        "output_json_path": "out/map.json",
        "batch_size": 250,
        "top_n": 12,
        "min_coverage": 0.87
    }
}
""".strip(),
        encoding="utf-8",
    )

    args = _parse_args(
        [
            "--config",
            str(config_path),
            "override.parquet",
            "override.json",
            "--batch-size",
            "500",
            "--top-n",
            "5",
            "--min-coverage",
            "0.9",
        ]
    )

    assert args.parquet_path == "override.parquet"
    assert args.output_json_path == "override.json"
    assert args.batch_size == 500
    assert args.top_n == 5
    assert args.min_coverage == 0.9
