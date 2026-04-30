"""Unit tests for BaseForLoopStrategy.

Validates the concrete node deletion utility and ensures the abstract 
interface for loop component extraction is correctly defined.
"""

import pytest
from unittest.mock import MagicMock
from typing import List

from transtructiver.node import Node
from transtructiver.mutation.mutation_context import MutationContext
from transtructiver.mutation.rules.mutation_rule import MutationRecord, MutationRule
from transtructiver.mutation.rules.control_structure_substitution.control_structure_strategies.for_loop_strategies.base_for_loop_strategy import (
    BaseForLoopStrategy,
)

# ===== Mock Implementation =====


class ConcreteForLoopStrategy(BaseForLoopStrategy):
    """Concrete implementation for testing BaseForLoopStrategy's shared logic."""

    def is_valid(self, node: Node) -> bool:
        return True

    def apply(self, node: Node, context: MutationContext, indent_unit: str) -> List[MutationRecord]:
        return []

    def _extract_for_loop_components(self, node: Node):
        """Mock implementation of abstract component extraction."""
        return ("init", "cond", "update", "body")

    def _clean_for_loop_header(self, node: Node, rule) -> List[MutationRecord]:
        """Mock implementation of abstract header cleaning."""
        return []


# ===== Helpers =====


def make_node(node_type: str = "test", parent: Node | None = None) -> Node:
    """Creates a node and optionally links it to a parent."""
    n = Node(start_point=(0, 0), end_point=(0, 1), type=node_type)
    if parent:
        n.parent = parent
        parent.children.append(n)
    return n


# ===== Fixtures =====


@pytest.fixture
def strategy():
    """Returns an instance of the concrete for-loop strategy."""
    return ConcreteForLoopStrategy()


@pytest.fixture
def mock_rule():
    """Returns a mock MutationRule."""
    return MagicMock(spec=MutationRule)


# ===== Test Cases =====


class TestBaseForLoopStrategyLogic:
    """Tests for concrete methods in the BaseForLoopStrategy class."""

    def test_delete_nodes_generates_records(self, strategy, mock_rule):
        """
        Ensures _delete_nodes calls rule.record_delete for every provided node
        and returns the collected records.
        """
        parent = make_node("parent")
        node_a = make_node("node_a", parent=parent)
        node_b = make_node("node_b", parent=parent)

        # Setup mock return values for records
        record_a = MagicMock(spec=MutationRecord)
        record_b = MagicMock(spec=MutationRecord)
        mock_rule.record_delete.side_effect = [record_a, record_b]

        nodes_to_delete = [node_a, node_b]
        records = strategy._delete_nodes(nodes_to_delete, mock_rule)

        # Assertions
        assert len(records) == 2
        assert records == [record_a, record_b]

        # Verify rule was called correctly for both
        mock_rule.record_delete.assert_any_call(parent, node_a)
        mock_rule.record_delete.assert_any_call(parent, node_b)

    def test_delete_nodes_empty_list(self, strategy, mock_rule):
        """Ensures an empty list of nodes returns an empty list of records."""
        assert strategy._delete_nodes([], mock_rule) == []


class TestBaseForLoopStrategyInterface:
    """Tests ensuring the abstract contract is enforced."""

    def test_cannot_instantiate_without_abstract_methods(self):
        """Verifies that the class remains abstract until all methods are implemented."""

        class IncompleteStrategy(BaseForLoopStrategy):
            def is_valid(self, node):
                return True

            def apply(self, node, ctx, indent):
                return []

            # Missing _extract_for_loop_components and _clean_for_loop_header

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStrategy() # type: ignore

    def test_extract_components_contract(self, strategy):
        """Verifies that the implemented extract method follows the intended signature."""
        dummy_node = make_node()
        components = strategy._extract_for_loop_components(dummy_node)

        assert isinstance(components, tuple)
        assert len(components) == 4
        assert components[0] == "init"
