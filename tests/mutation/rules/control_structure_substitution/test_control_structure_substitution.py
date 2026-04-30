"""Unit tests for the ControlStructureSubstitutionRule mutation rule.

Validates the rule's ability to coordinate strategy discovery, 
manage identifier tracking (taken_names), and apply transformations.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List

from transtructiver.mutation.mutation_types import MutationAction
from transtructiver.mutation.rules.control_structure_substitution.control_structure_substitution import (
    ControlStructureSubstitutionRule,
)
from transtructiver.node import Node
from transtructiver.mutation.mutation_context import MutationContext
from transtructiver.mutation.rules.mutation_rule import MutationRecord

# ===== Helpers =====


def _wire_parents(node: Node, parent: Node | None = None) -> Node:
    """Recursively populate parent links for traversal testing."""
    node.parent = parent
    for child in node.children:
        _wire_parents(child, node)
    return node


def make_simple_python_tree(identifiers: List[str] | None = None, has_for: bool = False) -> Node:
    """Create a basic tree structure for testing traversal and identifier collection."""
    children = []
    if identifiers:
        for i, name in enumerate(identifiers):
            children.append(Node((i, 0), (i, len(name)), "identifier", text=name))

    if has_for:
        # Simplified for_statement node
        children.append(Node((10, 0), (12, 0), "for_statement", text="for x in y: pass"))

    root = Node((0, 0), (20, 0), "module", children=children)
    root.language = "python"
    return _wire_parents(root)


# ===== Fixtures =====


@pytest.fixture
def mutation_context():
    """Return a default MutationContext for testing."""
    return MutationContext()


@pytest.fixture
def rule():
    """Return an instance of the ControlStructureSubstitutionRule."""
    return ControlStructureSubstitutionRule()


# ===== Test Cases =====


class TestControlStructureSubstitutionCore:
    """Core functionality tests for rule initialization and error handling."""

    def test_rule_initialization(self, rule):
        """Ensure the rule identifies itself correctly."""
        assert rule.rule_name == "control-structure-substitution"

    def test_apply_with_none_root(self, rule, mutation_context):
        """Ensure apply handles None root gracefully."""
        assert rule.apply(None, mutation_context) == []

    def test_missing_language_raises_value_error(self, rule, mutation_context):
        """The rule must raise an error if the root has no language attribute."""
        root = Node((0, 0), (1, 0), "module")
        root.language = None
        with pytest.raises(ValueError, match="No language found on root node"):
            rule.apply(root, mutation_context)


class TestIdentifierCollection:
    """Tests for the identifier tracking (taken_names) logic."""

    def test_collects_identifiers_into_context(self, rule, mutation_context):
        """Ensure the rule populates context.taken_names from all identifiers in the tree."""
        names = ["var_a", "func_b", "iter_var"]
        root = make_simple_python_tree(identifiers=names)

        rule.apply(root, mutation_context)

        assert hasattr(mutation_context, "taken_names")
        for name in names:
            assert name in mutation_context.taken_names

    def test_taken_names_is_a_set(self, rule, mutation_context):
        """Verify taken_names is a set to ensure unique collection and O(1) lookups."""
        root = make_simple_python_tree(identifiers=["x", "x", "y"])
        rule.apply(root, mutation_context)

        assert isinstance(mutation_context.taken_names, set)
        assert len(mutation_context.taken_names) == 2


class TestStrategyCoordination:
    """Tests verifying that the rule correctly delegates to strategies."""

    @patch(
        "transtructiver.mutation.rules.control_structure_substitution.control_structure_substitution.get_for_loop_strategy"
    )
    def test_applies_strategy_on_valid_node(self, mock_get_strategy, rule, mutation_context):
        """
        Verify that the rule detects a valid node and calls apply on the appropriate strategy.
        """
        # Setup Mock Strategy
        mock_strategy = MagicMock()
        mock_strategy.is_valid.side_effect = lambda n: n.type == "for_statement"
        mock_strategy.apply.return_value = [MutationRecord((10, 0), MagicMock(), metadata={})]
        mock_get_strategy.return_value = mock_strategy

        # Setup Tree
        root = make_simple_python_tree(has_for=True)

        records = rule.apply(root, mutation_context)

        # Assertions
        assert len(records) == 1
        assert mock_strategy.apply.called
        # Check that context.taken_names was passed through
        args, _ = mock_strategy.apply.call_args
        assert args[2] == mutation_context

    def test_ignores_non_loop_nodes(self, rule, mutation_context):
        """The rule should return no records if no control structures are found."""
        root = make_simple_python_tree(identifiers=["a", "b"])
        # Language is python, strategy is for_loop, but tree has no for_statement
        records = rule.apply(root, mutation_context)

        assert records == []


class TestIndentationHandling:
    """Tests for proper indentation detection during rule application."""

    @patch(
        "transtructiver.mutation.rules.control_structure_substitution.control_structure_substitution.IndentationUtils"
    )
    def test_detects_and_passes_indent_unit(self, mock_indent_utils, rule, mutation_context):
        """Ensure the rule detects indentation and passes it to the strategy."""
        mock_indent_utils.detect_indent_unit.return_value = "    "

        mock_strategy = MagicMock()
        mock_strategy.is_valid.return_value = True
        mock_strategy.apply.return_value = []

        with patch(
            "transtructiver.mutation.rules.control_structure_substitution.control_structure_substitution.get_for_loop_strategy",
            return_value=mock_strategy,
        ):
            root = make_simple_python_tree(has_for=True)
            rule.apply(root, mutation_context)

            # Check if detect_indent_unit was called on root
            mock_indent_utils.detect_indent_unit.assert_called_with(root)
            # Check if the indent_unit "    " was passed to strategy.apply
            args, _ = mock_strategy.apply.call_args
            assert args[3] == "    "
