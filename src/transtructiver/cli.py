"""Command-line interface for the TranStructIVer pipeline.

This module provides the entry point for running the complete transformation
pipeline on datasets, including data loading, parsing, mutation, and verification.

Outputs written per run (inside ``--output-dir``):
    * ``manifest.jsonl``       — one JSON-Lines record per snippet
    * ``augmented_dataset.parquet`` — original/mutated code pairs
    * ``summary_log.csv``      — semantics-preservation pass/fail log
"""

# Python version precheck
import sys
import itertools
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

if sys.version_info < (3, 14):
    sys.exit("Error: Python 3.14 or higher is required. Please upgrade your interpreter.")

import argparse
import importlib
import inspect
import json
import os
import pkgutil
import re
from dataclasses import dataclass
from .data_loading.data_loader import DataLoader
from .parsing.parser import Parser
from .config import load_config, resolve_enabled_rules, get_rule_params
from .mutation.mutation_engine import MutationEngine
from .mutation.rules.identifier_renaming.rename_identifiers import RenameIdentifiersRule
from .mutation.rules.mutation_rule import MutationRule
from .node import Node
from .reporting import summary_logger
from .reporting.output_manager import OutputManager, RunStats
from .verification.si_verifier import SIVerifier

####################################################################
# Rule registry and discovery
####################################################################


def _class_to_rule_name(class_name: str) -> str:
    """Convert a CamelCase class name to a kebab-case rule name.

    Args:
        class_name (str): The class name to convert.

    Returns:
        str: The kebab-case rule name.

    Examples:
        CommentDeletionRule  → comment-deletion
        WhitespaceNormalizationRule → whitespace-normalization
    """
    if class_name.endswith("Rule"):
        class_name = class_name[:-4]
    return re.sub(r"(?<!^)(?=[A-Z])", "-", class_name).lower()


def _build_rule_registry() -> dict[str, type[MutationRule]]:
    """Auto-discover all MutationRule subclasses in the mutation/rules package.

    Returns:
            dict[str, type[MutationRule]]: Registry mapping rule names to classes.

    Each rule class is registered under:
    - Its explicit ``rule_name`` class attribute (if defined), OR
    - A name auto-derived from its class name (CamelCase → kebab-case,
        trailing 'Rule' stripped).

    New rules are picked up automatically when their module is placed anywhere
    inside ``transtructiver/mutation/rules/``.  No manual registration needed.
    """
    import transtructiver.mutation.rules as rules_pkg

    registry: dict[str, type[MutationRule]] = {}

    for _finder, module_name, _is_pkg in pkgutil.walk_packages(
        path=rules_pkg.__path__,
        prefix=rules_pkg.__name__ + ".",
        onerror=lambda _: None,
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for _attr_name, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, MutationRule)
                and cls is not MutationRule
                and cls.__module__ == module.__name__
            ):
                rule_key = getattr(cls, "rule_name", None) or _class_to_rule_name(cls.__name__)
                registry[rule_key] = cls

    return registry


RULE_REGISTRY: dict[str, type[MutationRule]] = _build_rule_registry()


def _pipeline_worker_task(batch_items, rules, rule_params, verifier_options_dict):
    """Worker task for processing a batch of snippets in parallel."""
    # Lazy initialize components per process to avoid overhead if process is reused
    global _worker_engine, _worker_parser, _worker_verifier
    if "_worker_engine" not in globals():
        _worker_engine = _build_engine(rules, rule_params)
        _worker_parser = Parser()
        _worker_verifier = SIVerifier(
            strictness=verifier_options_dict.get("strictness", "strict"),
            max_errors=verifier_options_dict.get("max_errors"),
        )

    results = []
    cst_pairs = []  # (orig, mut)

    for item in batch_items:
        code = item["code"]
        lang = item["language"]

        orig_cst, _ = _worker_parser.parse(code, lang)

        if orig_cst is None:
            results.append({"idx": item["idx"], "skipped": True})
            cst_pairs.append(None)
        else:
            mut_cst = orig_cst.clone()
            cst_pairs.append((orig_cst, mut_cst))

    valid_mut_csts = [pair[1] for pair in cst_pairs if pair is not None]
    manifests = _worker_engine.apply_mutations_batch(valid_mut_csts) if valid_mut_csts else []

    m_idx = 0
    for i, item in enumerate(batch_items):
        if cst_pairs[i] is None:
            continue
        orig, mut = cst_pairs[i]
        manifest = manifests[m_idx]
        m_idx += 1
        verified = _worker_verifier.verify(orig, mut, manifest)
        results.append(
            {
                "idx": item["idx"],
                "snippet_id": f"row_{item['idx']}",
                "skipped": False,
                "manifest_dict": manifest.to_dict(),
                "is_manifest_empty": manifest.is_empty(),
                "original_code": orig.to_code(),
                "mutated_code": mut.to_code(),
                "mutated_cst_json": mut.to_json(),
                "verified": verified,
                "errors": list(_worker_verifier.errors),
                "row": item["row"],
            }
        )
    return results


_RULE_PARAM_PROPAGATIONS = {
    "rename-identifier": {
        "control-structure-substitution": {
            "level": lambda v: v,
        }
    },
    "whitespace-normalization": {
        "dead-code-insertion": {
            "indent_unit": lambda base_unit: " " * base_unit,
        }
    },
}

####################################################################
# Dataclasses and prototype helpers
####################################################################

# PROTOTYPE-ONLY OUTPUT:
# Keep developer-facing logs enabled in prototype runs.
# For production hardening, set this to False (or remove related helpers/calls).
PROTOTYPE_OUTPUT_ENABLED = False


@dataclass
class VerifierOptions:
    """Runtime options for semantic-isomorphism verification strictness."""

    strictness: str = "strict"
    max_errors: int | None = None


@dataclass
class PipelineOptions:
    """Runtime options for large-scale dataset processing."""

    batch_size: int = 1000
    checkpoint_every: int = 1000
    checkpoint_path: str = "output/checkpoint.json"
    resume: bool = False
    max_rows_per_shard: int = 0
    compress_output: bool = False
    workers: int = 1


def _prototype_log(message: str) -> None:
    """PROTOTYPE-ONLY: central log sink for easy production removal.

    Args:
        message (str): The log message to print.
    """
    if PROTOTYPE_OUTPUT_ENABLED:
        print(message)


def _prototype_pretty(label: str, root: Node) -> None:
    """PROTOTYPE-ONLY: pretty-print a CST for debugging and demos.

    Args:
        label (str): Label to print before the CST.
        root (Node): CST root node to pretty-print.
    """
    if PROTOTYPE_OUTPUT_ENABLED:
        print(label)
        print(root.to_code())


####################################################################
# Rule validation and engine construction
####################################################################


def _validate_rules(rules: list[str]) -> list[str]:
    """Return unsupported rule names so callers can fail fast and explicitly.

    Args:
        rules (list[str]): List of rule names to validate.

    Returns:
        list[str]: List of unsupported rule names.
    """
    return [rule for rule in rules if rule not in RULE_REGISTRY]


def _build_engine(
    rules: list[str],
    rule_params: dict[str, dict] | None = None,
) -> MutationEngine:
    """Construct a mutation engine with per-rule configuration.

    Args:
        rules (list[str]): List of rule names to apply.
        rule_params (dict[str, dict] | None): Per-rule parameters.

    Returns:
        MutationEngine: Configured mutation engine.
    """
    rule_params = rule_params or {}
    configured_rules = []

    # ------------------------------------------------------------
    # Parameter propagation between rules (generic system)
    # ------------------------------------------------------------
    for src_rule, targets in _RULE_PARAM_PROPAGATIONS.items():
        if src_rule not in rule_params:
            continue

        src_params = rule_params.get(src_rule, {})

        for target_rule, mappings in targets.items():
            if target_rule not in rules:
                continue

            target_params = rule_params.setdefault(target_rule, {})

            for key, transform in mappings.items():
                if key not in src_params:
                    continue

                # do not override explicit CLI/user values
                if key in target_params:
                    continue

                target_params[key] = transform(src_params[key])

    # ------------------------------------------------------------
    # Build rule instances
    # ------------------------------------------------------------
    for rule_name in rules:
        rule_cls = RULE_REGISTRY[rule_name]
        params = dict(rule_params.get(rule_name) or {})

        try:
            configured_rules.append(rule_cls(**params) if params else rule_cls())
        except TypeError as exc:
            raise ValueError(
                f"Invalid parameters for rule '{rule_name}': {params}. Error: {exc}"
            ) from exc

    return MutationEngine(configured_rules)


####################################################################
# Pipeline execution
####################################################################


def run_pipeline(
    filepath: str,
    rules: list[str],
    output_dir: str = "output",
    rule_params: dict[str, dict] | None = None,
    pipeline_options: PipelineOptions | None = None,
    verifier_options: VerifierOptions | None = None,
):
    """Run the complete TranStructIVer pipeline on a dataset file.

    For each snippet the pipeline:
        1. Parses the original code into a CST.
        2. Clones the CST and applies mutation rules.
        3. Writes the transformation manifest.
        4. Writes the original/mutated code pair to the augmented dataset.
        5. Verifies semantic preservation and logs the result.

    Args:
        filepath (str): Path to the dataset file (Parquet format).
        rules (list[str]): Mutation rule names to apply (see RULE_REGISTRY).
        output_dir (str): Directory for all output files. Created if absent.
        rule_params (dict[str, dict] | None): Parameters for rule configuration
            (e.g., level and targets for RenameIdentifiersRule).
        pipeline_options (PipelineOptions | None): Performance options for
            streaming, sharding, compression, and checkpointing.
        verifier_options (VerifierOptions | None): Options for auditor thresholds and
            strictness levels for verification.

    Raises:
        ValueError: If any rule name is not registered in RULE_REGISTRY.
    """
    os.makedirs(output_dir, exist_ok=True)
    pipeline_options = pipeline_options or PipelineOptions()
    verifier_options = verifier_options or VerifierOptions()
    # Use DataLoader abstraction; implementation is chosen internally.
    loader = DataLoader(filepath, checkpoint_path=pipeline_options.checkpoint_path)
    start_index = loader.load_checkpoint(pipeline_options.resume)

    parser = Parser()

    unsupported_rules = _validate_rules(rules)
    if unsupported_rules:
        raise ValueError(f"Arguments contain unsupported mutation rule: {unsupported_rules}")

    # Initialize engine once to inspect rules and for use in sequential path
    engine = _build_engine(rules, rule_params)

    # Find RenameIdentifiersRule to determine if we need to batch snippets (Level 1 MLM)
    rename_rule = next(
        (
            rule
            for rule in engine.rules
            if isinstance(rule, RenameIdentifiersRule) and rule.level == 1
        ),
        None,
    )
    batch_snippets = rename_rule.batch_snippets if rename_rule else 0

    stats = RunStats()
    processed_since_checkpoint = 0

    with OutputManager(
        output_dir,
        max_rows_per_shard=pipeline_options.max_rows_per_shard,
        compress_output=pipeline_options.compress_output,
    ) as outputs:
        index = start_index

        def handle_res(res):
            nonlocal processed_since_checkpoint, index
            idx = res["idx"]
            index = idx
            if res.get("skipped"):
                stats.parse_skipped += 1
            else:
                stats.parsed_ok += 1
                snippet_id = res["snippet_id"]
                outputs.write_manifest(idx, snippet_id, res["manifest_dict"])

                row = res["row"]
                metadata = dict(row)
                for key in [
                    "code",
                    "language",
                    "label",
                    "mutated_cst",
                    "mutated_code",
                    "original_code",
                ]:
                    metadata.pop(key, None)

                outputs.write_dataset_row(
                    idx,
                    snippet_id,
                    res["original_code"],
                    res["mutated_code"],
                    row.get("language"),
                    row.get("label"),
                    has_mutation_applied=not res["is_manifest_empty"],
                    metadata=metadata,
                    mutated_cst=Node.from_json(res["mutated_cst_json"]),
                )

                if res["verified"]:
                    stats.verified_ok += 1
                else:
                    stats.verified_fail += 1

                summary_logger.write_summary(
                    snippet_id=snippet_id,
                    verified=res["verified"],
                    errors=res["errors"],
                    writer=outputs.summary_writer,
                )

            processed_since_checkpoint += 1
            if (
                pipeline_options.checkpoint_every > 0
                and processed_since_checkpoint >= pipeline_options.checkpoint_every
            ):
                loader.save_checkpoint(idx + 1, stats)
                processed_since_checkpoint = 0

        snippets = loader.iter_snippets(
            batch_size=pipeline_options.batch_size,
            start_index=start_index,
        )
        v_opt = {
            "strictness": verifier_options.strictness,
            "max_errors": verifier_options.max_errors,
        }

        worker_batch_size = max(batch_snippets, 50) if batch_snippets > 0 else 100

        def batch_generator():
            """Yields contiguous batches of snippet data for parallel or sequential processing."""
            current_batch = []
            for idx, row in snippets:
                current_batch.append(
                    {
                        "idx": idx,
                        "language": row.get("language"),
                        "code": row.get("code") or row.get("mutated_code"),
                        "mutated_cst_json": row.get("mutated_cst"),
                        "row": row,
                    }
                )
                if len(current_batch) >= worker_batch_size:
                    yield current_batch
                    current_batch = []
            if current_batch:
                yield current_batch

        if pipeline_options.workers > 1:
            # Process batches in parallel using order-preserving map
            worker_fn = functools.partial(
                _pipeline_worker_task,
                rules=rules,
                rule_params=rule_params,
                verifier_options_dict=v_opt,
            )
            with ProcessPoolExecutor(max_workers=pipeline_options.workers) as executor:
                total_batches = (
                    loader.num_rows - start_index + worker_batch_size - 1
                ) // worker_batch_size
                for batch_results in tqdm(
                    executor.map(worker_fn, batch_generator()),
                    total=total_batches,
                    desc="Processing Batches",
                ):
                    for res in batch_results:
                        handle_res(res)
        else:
            # Sequential processing using the same task logic for consistency
            total_batches = (
                loader.num_rows - start_index + worker_batch_size - 1
            ) // worker_batch_size
            for batch_items in tqdm(
                batch_generator(), total=total_batches, desc="Processing Batches"
            ):
                for res in _pipeline_worker_task(batch_items, rules, rule_params, v_opt):
                    handle_res(res)

        if processed_since_checkpoint > 0:
            loader.save_checkpoint(index + 1, stats)

        summary_logger.write_summary_totals(
            parsed_ok=stats.parsed_ok,
            parse_skipped=stats.parse_skipped,
            verified_ok=stats.verified_ok,
            verified_fail=stats.verified_fail,
            writer=outputs.summary_writer,
        )

        manifest_path, dataset_path, summary_path = outputs.output_paths_summary()


####################################################################
# CLI argument parsing and main entry
####################################################################


def main():
    """Main entry point for the TranStructIVer CLI.

    Command-line Arguments:
        filepath: Path to the dataset file (Parquet format) to process.
        --rules: Mutation rules to apply (default: rename-identifier).
        --output-dir: Directory for output files (default: output).

    Example:
        uv run cli src\\transtructiver\\prototype\\data_load\\sample.parquet
        uv run cli dataset.parquet --rules rename-identifier --output-dir results
    """
    argparser = argparse.ArgumentParser(
        prog="TranStructIVer", description="Run the TranStructIVer pipeline on a dataset file."
    )
    argparser.add_argument("filepath", help="Path to the dataset file")
    argparser.add_argument("rules", nargs="*", help="Mutation rules", default=None)
    argparser.add_argument(
        "--config",
        default=None,
        help="Path to external YAML config file.",
    )
    argparser.add_argument(
        "--output-dir", default=None, help="Directory for output files (default: output)"
    )
    argparser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Parquet streaming batch size (default: 1000)",
    )
    argparser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume processing from checkpoint if checkpoint file exists.",
    )
    argparser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Checkpoint path (default: <output-dir>/checkpoint.json)",
    )
    argparser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Write checkpoint every N processed snippets (default: 1000)",
    )
    argparser.add_argument(
        "--max-rows-per-shard",
        type=int,
        default=None,
        help="Shard manifest/dataset outputs every N rows (0 disables sharding).",
    )
    argparser.add_argument(
        "--compress-output",
        action="store_true",
        default=None,
        help="Compress output files using gzip.",
    )
    argparser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: 1)",
    )
    argparser.add_argument(
        "--rule-param",
        action="append",
        default=[],
        metavar="RULE:PARAM=VALUE",
        help="Specify a rule parameter as rule:param=value (can be repeated)",
    )
    argparser.add_argument(
        "--verifier-strictness",
        choices=["strict", "balanced", "lenient"],
        default=None,
        help="Verification strictness level (FR-8.2).",
    )
    argparser.add_argument(
        "--verifier-max-errors",
        type=int,
        default=None,
        help="Maximum tolerated verification errors before snippet fails (FR-8.2).",
    )
    args = argparser.parse_args()

    config = load_config(args.config)

    rules = resolve_enabled_rules(config, args.rules)

    execution_cfg = config.execution

    def _coalesce(cli_value, key: str, default):
        return cli_value if cli_value is not None else execution_cfg.get(key, default)

    output_dir = _coalesce(args.output_dir, "output_dir", "output")

    verifier_options = VerifierOptions(
        strictness=(args.verifier_strictness or config.verifier.strictness or "strict"),
        max_errors=(
            args.verifier_max_errors
            if args.verifier_max_errors is not None
            else config.verifier.max_errors
        ),
    )

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")

    pipeline_options = PipelineOptions(
        batch_size=_coalesce(args.batch_size, "batch_size", 1000),
        checkpoint_every=_coalesce(args.checkpoint_every, "checkpoint_every", 1000),
        checkpoint_path=checkpoint_path,
        resume=_coalesce(args.resume, "resume", False),
        max_rows_per_shard=_coalesce(args.max_rows_per_shard, "max_rows_per_shard", 0),
        compress_output=_coalesce(args.compress_output, "compress_output", False),
        workers=_coalesce(args.workers, "workers", 1),
    )

    # Start with config-based params
    rule_params_map = {rule_name: get_rule_params(config, rule_name) for rule_name in rules}

    # Parse --rule-param CLI overrides (rule:param=value)
    for param in args.rule_param:
        # Accept rule:param=value or rule:param=json_value
        if ":" not in param or "=" not in param:
            raise ValueError(f"Invalid --rule-param format: {param}. Use rule:param=value")
        rule_key, rest = param.split(":", 1)
        param_key, value = rest.split("=", 1)
        # Try to parse value as JSON, fallback to string
        try:
            parsed_value = json.loads(value)
        except Exception:
            parsed_value = value
        if rule_key not in rule_params_map:
            rule_params_map[rule_key] = {}
        rule_params_map[rule_key][param_key] = parsed_value

    run_pipeline(
        args.filepath, rules, output_dir, rule_params_map, pipeline_options, verifier_options
    )


if __name__ == "__main__":
    main()
