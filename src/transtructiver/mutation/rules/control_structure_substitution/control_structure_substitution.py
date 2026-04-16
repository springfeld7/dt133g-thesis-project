"""
Rule for transforming control structures into alternative forms.

Currently supports transformation of 'for' loops into 'while' loops,
but is designed to support additional control structures (e.g., ternary)
in the future.
"""

from typing import List

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

        language = root.language.lower().strip() if root.language else None
        if not language:
            raise ValueError("No language found on root node.")

        strategies: List[BaseControlStructureStrategy] = [
            get_for_loop_strategy(language),
        ]

        targets: List[tuple[BaseControlStructureStrategy, Node]] = []

        indent_unit = IndentationUtils.detect_indent_unit(root)

        for node in root.traverse():
            # Collect valid targets for transformation
            for strategy in strategies:
                if strategy.is_valid(node):
                    targets.append((strategy, node))

        # Apply transformations
        for strategy, node in targets:
            records.extend(strategy.apply(node, self, context, indent_unit))

        return records
