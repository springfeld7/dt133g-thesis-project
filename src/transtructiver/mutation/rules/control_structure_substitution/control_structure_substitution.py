"""
Rule for transforming control structures into alternative forms.

Currently supports transformation of 'for' loops into 'while' loops,
but is designed to support additional control structures (e.g., ternary)
in the future.
"""

import random
from typing import Dict, List

from ....node import Node
from ..mutation_rule import MutationRule, MutationRecord
from ...mutation_context import MutationContext
from ..utils.indentation_util import IndentationUtils
from .control_structure_strategies.for_loop_strategies.registry import get_for_loop_strategy
from .control_structure_strategies.base_control_structure_strategy import (
    BaseControlStructureStrategy,
)


class ControlStructureSubstitutionRule(MutationRule):
    """
    Rewrites eligible control structures into alternative forms.

    Attributes:
        rule_name (str): CLI identifier for the rule.
    """

    rule_name = "control-structure-substitution"

    # Mapping of transformation levels to the percentage of applicable targets to transform.
    LEVEL_RATIOS: Dict[int, float] = {
        0: 0.10,  # Minimal: 10% of targets
        1: 0.35,  # Low: 35% of targets
        2: 0.65,  # Medium: 65% of targets
        3: 1.00,  # Maximum: 100% of targets
    }

    def __init__(self, level: int = 0, seed: int = 42):
        """
        Initialize the rule with a specified transformation level passed to rename-identifiers rule,"
        "or 0 for default.

        Args:
            level (int): The transformation level to apply, influencing the aggressiveness of renaming.
        """
        super().__init__()
        self._level = level
        self._rng = random.Random(seed)

    def apply(self, root: Node, context: MutationContext) -> List[MutationRecord]:
        """
        Entry point for the mutation rule.

        Traverses the CST and applies all applicable control structure strategies.

        Args:
            root (Node): The root of the CST.
            context (MutationContext): Context for mutation tracking.

        Returns:
            List[MutationRecord]: Records of all transformations applied.
        """
        if root is None:
            return []

        records: List[MutationRecord] = []
        taken_names = set()

        language = root.language.lower().strip() if root.language else None
        if not language:
            raise ValueError("No language found on root node.")

        strategies: List[BaseControlStructureStrategy] = [
            get_for_loop_strategy(language),
        ]

        targets: List[tuple[BaseControlStructureStrategy, Node]] = []
        indent_unit = IndentationUtils.detect_indent_unit(root)

        for node in root.traverse():

            # Collect variable names to avoid naming collisions in transformations
            if node.type == "identifier":
                taken_names.add(node.text)

            # Collect valid targets for transformation
            for strategy in strategies:
                if strategy.is_valid(node):
                    targets.append((strategy, node))

        context.taken_names = taken_names

        if not targets:
            return []

        ratio = self.LEVEL_RATIOS.get(self._level, 0.15)
        num_to_transform = max(1, round(len(targets) * ratio))
        selected_targets = self._rng.sample(targets, num_to_transform)

        # Apply transformations
        for strategy, node in selected_targets:
            records.extend(strategy.apply(node, self, context, indent_unit, self._level))

        return records
