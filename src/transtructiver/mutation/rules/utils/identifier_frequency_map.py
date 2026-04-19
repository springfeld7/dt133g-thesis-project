"""Build compact identifier-frequency maps from human-written source samples.

This module streams snippets from a parquet dataset, parses them through the
existing parser/annotation pipeline, and aggregates identifier frequencies by
semantic label (for example ``variable_name`` and ``parameter_name``).

The resulting JSON is optimized for one-pass loading at runtime:
    - top-level keys are semantic labels
    - nested keys are identifier strings
    - values are observed frequencies
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generator

from ....config import _load_yaml, _load_toml
from ....data_loading.data_loader import DataLoader
from ....node import Node
from ....parsing.parser import Parser


IdentifierCounterMap = dict[str, Counter[str]]
RoleTypeMap = dict[str, IdentifierCounterMap]


@dataclass(slots=True)
class BuildStats:
    """Summary statistics for a frequency-map build run."""

    rows_seen: int = 0
    rows_parsed: int = 0
    rows_skipped: int = 0
    identifiers_counted: int = 0


def _iter_role_identifiers(root: Node) -> Generator[tuple[str, str, str | None], Any, None]:
    """Yield (semantic_label, identifier_text) pairs from an annotated tree."""
    for node in root.traverse():
        if not node.text or "identifier" not in node.type:
            continue
        if node.builtin:
            continue
        if len(node.text) < 3:
            continue
        if not node.semantic_label or not node.semantic_label.endswith("_name"):
            continue
        yield node.semantic_label, node.text, node.context_type


def collect_identifier_frequencies(
    parquet_path: str,
    batch_size: int = 1000,
) -> tuple[dict[str, RoleTypeMap], BuildStats]:
    """Collect identifier frequencies grouped by semantic label.

    Args:
        parquet_path: Path to input parquet with ``code`` and ``language`` columns.
        batch_size: Streaming parquet batch size.

    Returns:
        A tuple of (role_counters, stats).
    """
    parser = Parser()
    loader = DataLoader(parquet_path)

    stats = BuildStats()
    language_role_map: dict[str, RoleTypeMap] = {}

    for _idx, code, language in loader.iter_snippets(batch_size=batch_size, start_index=0):
        stats.rows_seen += 1

        root, parse_error = parser.parse(code, language)
        if root is None:
            if parse_error:
                stats.rows_skipped += 1
            continue

        stats.rows_parsed += 1
        role_type_map: RoleTypeMap = language_role_map.setdefault(language, {})

        for role, identifier, context_type in _iter_role_identifiers(root):
            ct = context_type if context_type else "none"
            type_counters = role_type_map.setdefault(role, {})
            type_counters.setdefault(ct, Counter())[identifier] += 1

            stats.identifiers_counted += 1

    print(f"Collected id frequencies for {stats.rows_seen} samples")
    return language_role_map, stats


def _select_top_identifiers(
    counter: Counter[str],
    top_n: int | None,
    min_coverage: float,
) -> list[tuple[str, int]]:
    """Select top identifiers either by fixed N or by cumulative coverage."""
    if not counter:
        return []

    ranked = counter.most_common()
    if top_n is not None:
        return ranked[:top_n]

    total = sum(counter.values())
    running = 0
    selected: list[tuple[str, int]] = []
    for identifier, count in ranked:
        selected.append((identifier, count))
        running += count
        if total > 0 and (running / total) >= min_coverage:
            break

    return selected


def build_frequency_payload(
    roles_by_language: dict[str, RoleTypeMap],
    stats: BuildStats,
    source_parquet: str,
    top_n: int | None = None,
    min_coverage: float = 0.95,
) -> dict[str, Any]:
    """Build the JSON-serializable payload for identifier distribution maps."""
    language_role_map: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    for language in sorted(roles_by_language.keys()):
        role_types = roles_by_language[language]
        language_role_map[language] = {}

        for role in sorted(role_types.keys()):
            type_counters = role_types[role]
            role_entries: dict[str, dict[str, int]] = {}

            for type_name in sorted(type_counters.keys()):
                counter = type_counters[type_name]
                selected = _select_top_identifiers(counter, top_n=top_n, min_coverage=min_coverage)
                role_entries[type_name] = {name: count for name, count in selected}

            # If no types, still add empty dict for role
            if not role_entries:
                role_entries = {}

            language_role_map[language][role] = role_entries

    payload = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection": {
            "top_n": top_n,
            "min_coverage": min_coverage,
            "mode": "fixed-top-n" if top_n is not None else "coverage-auto",
        },
        "source": {
            "parquet_path": source_parquet,
            "rows_seen": stats.rows_seen,
            "rows_parsed": stats.rows_parsed,
            "rows_skipped": stats.rows_skipped,
            "identifiers_counted": stats.identifiers_counted,
        },
        "role_maps": language_role_map,
    }

    return payload


def build_identifier_frequency_map(
    parquet_path: str,
    output_json_path: str,
    *,
    batch_size: int = 1000,
    top_n: int | None = None,
    min_coverage: float = 0.95,
) -> dict:
    """Build and write a role-keyed identifier frequency map JSON."""
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be positive when provided")
    if not (0.0 < min_coverage <= 1.0):
        raise ValueError("min_coverage must be in the range (0.0, 1.0]")

    roles_by_language, stats = collect_identifier_frequencies(parquet_path, batch_size=batch_size)
    payload = build_frequency_payload(
        roles_by_language,
        stats,
        source_parquet=parquet_path,
        top_n=top_n,
        min_coverage=min_coverage,
    )

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    return payload


def load_identifier_frequency_map(path: str, language: str, role: str) -> dict[str, dict[str, int]]:
    """Load the generated JSON and return identifier -> count map."""
    with open(path, "r", encoding="utf-8") as fp:
        payload: dict[str, Any] = json.load(fp)
        role_maps: dict[str, dict[str, Any]] = payload.get("role_maps", {})
        language_role_map: dict[str, Any] = role_maps.get(language, {})
        role_entry = language_role_map.get(role, {})
        return role_entry


def _load_config_dict(path: str | None) -> dict[str, Any]:
    """Load config file contents, auto-detecting format by file extension."""
    if not path:
        return {}

    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data or {}
    if suffix == ".toml":
        return _load_toml(path)
    return _load_yaml(path)


def _extract_tool_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract tool-specific config section or fall back to top-level keys."""
    for key in ("identifier_frequency_map", "build_identifier_map"):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return raw


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build-identifier-map",
        description=(
            "Build a semantic-label keyed identifier frequency map from a parquet dataset."
        ),
    )
    parser.add_argument("parquet_path", nargs="?", default=None, help="Input parquet file")
    parser.add_argument("output_json_path", nargs="?", default=None, help="Output JSON file")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to JSON/TOML/YAML config file. "
            "Supports either top-level keys or an `identifier_frequency_map` section."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Parquet streaming batch size (default: 1000)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help=(
            "Keep only top-N identifiers per semantic label. "
            "If omitted, N is selected automatically from --min-coverage."
        ),
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=None,
        help=(
            "When --top-n is omitted, keep enough identifiers per role to reach "
            "this cumulative frequency coverage (default: 0.95)."
        ),
    )
    args = parser.parse_args(argv)

    tool_cfg = _extract_tool_config(_load_config_dict(args.config))

    def _coalesce(value: Any, key: str, default: Any) -> Any:
        return value if value is not None else tool_cfg.get(key, default)

    args.parquet_path = _coalesce(args.parquet_path, "parquet_path", None)
    args.output_json_path = _coalesce(args.output_json_path, "output_json_path", None)
    args.batch_size = _coalesce(args.batch_size, "batch_size", 1000)
    args.top_n = _coalesce(args.top_n, "top_n", None)
    args.min_coverage = _coalesce(args.min_coverage, "min_coverage", 0.95)

    if not args.parquet_path:
        parser.error("Missing required `parquet_path` (CLI positional arg or config key).")
    if not args.output_json_path:
        parser.error("Missing required `output_json_path` (CLI positional arg or config key).")

    return args


def main() -> None:
    args = _parse_args()
    start = datetime.now()
    payload = build_identifier_frequency_map(
        args.parquet_path,
        args.output_json_path,
        batch_size=args.batch_size,
        top_n=args.top_n,
        min_coverage=args.min_coverage,
    )
    role_maps = payload.get("role_maps", {})
    role_count = sum(len(language_roles) for language_roles in role_maps.values())
    end = datetime.now()
    print(
        f"Wrote identifier frequency map with {role_count} role(s) to " f"{args.output_json_path}"
    )
    time_elapsed = (end - start).total_seconds()
    print(
        f"\nTime elapsed: {time_elapsed * 1e3:.3f} ms\n    {time_elapsed / 30000 * 1e3:.3f} per sample"
    )


if __name__ == "__main__":
    main()
