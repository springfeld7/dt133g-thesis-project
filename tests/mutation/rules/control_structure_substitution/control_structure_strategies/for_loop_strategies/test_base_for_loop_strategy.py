"""Unit tests for the BaseControlStructureStrategy.

Validates the shared utility methods for node insertion, substitution, 
and indentation detection used across control structure strategies.
"""

import pytest
from unittest.mock import MagicMock
from typing import List

from transtructiver.node import Node
from transtructiver.mutation.mutation_context import MutationContext
from transtructiver.mutation.rules.mutation_rule import MutationRecord, MutationRule
from transtructiver.mutation.rules.control_structure_substitution.control_structure_strategies.base_control_structure_strategy import (
    BaseControlStructureStrategy,
)

# ===== Mock Implementation =====


class ConcreteStrategy(BaseControlStructureStrategy):
    """Concrete subclass to test base methods since BaseControlStructureStrategy is abstract."""

    def is_valid(self, node: Node) -> bool:
        return True

    def apply(self, node: Node, context: MutationContext, indent_unit: str) -> List[MutationRecord]:
        return []


# ===== Helpers =====


def make_test_node(node_type="test", text="text", start=(1, 0), end=(1, 4)) -> Node:
    """Utility to create a Node instance for testing."""
    return Node(start_point=start, end_point=end, type=node_type, text=text)


# ===== Fixtures =====


@pytest.fixture
def strategy():
    """Returns an instance of the concrete strategy."""
    return ConcreteStrategy()


@pytest.fixture
def mock_context():
    """Returns a mock MutationContext with an ID counter."""
    ctx = MagicMock(spec=MutationContext)
    ctx._id_counter = 100

    def next_id():
        ctx._id_counter += 1
        return ctx._id_counter

    ctx.next_id.side_effect = next_id
    return ctx


@pytest.fixture
def mock_rule():
    """Returns a mock MutationRule."""
    return MagicMock(spec=MutationRule)


# ===== Test Cases =====


class TestBaseStrategyInsertion:
    """Tests for node and segment insertion helpers."""

    def test_insert_node(self, strategy, mock_context, mock_rule):
        """It should create a new node, attach it to parent, and record the insertion."""
        parent = make_test_node("block", "{\n}")
        parent.children = []
        point = (2, 4)

        record = strategy._insert_node(
            context=mock_context,
            parent=parent,
            point=point,
            type="statement",
            text="x = 10",
            index=0,
            rule=mock_rule,
        )

        # Structural validation
        assert len(parent.children) == 1
        inserted_node = parent.children[0]
        assert inserted_node.parent == parent
        assert inserted_node.type == "statement"
        assert inserted_node.text == "x = 10"
        assert inserted_node.start_point == (101, -1)

        # Record validation
        mock_rule.record_insert.assert_called_once_with(
            point=(101, -1), insertion_point=point, new_text="x = 10", new_type="statement"
        )
        assert record == mock_rule.record_insert.return_value

    def test_insert_segments_batch(self, strategy, mock_context, mock_rule):
        """
        It should insert multiple segments and return their records.
        Note: The current implementation inserts at a static index, causing
        segments to appear in reverse order if not handled carefully.
        """
        parent = make_test_node("block")
        parent.children = []
        segments = [("type_a", "content_a"), ("type_b", "content_b")]

        records = strategy._insert_segments(
            context=mock_context,
            parent=parent,
            point=(5, 0),
            segments=segments,
            index=0,
            rule=mock_rule,
        )

        assert len(parent.children) == 2
        assert len(records) == 2
        # Verify stack-like behavior (last in, first out at index 0)
        assert parent.children[0].text == "content_b"
        assert parent.children[1].text == "content_a"


class TestBaseStrategyMutation:
    """Tests for the substitution helper."""

    def test_substitute_updates_attributes(self, strategy, mock_rule):
        """It should update the node type and text and return a substitution record."""
        node = make_test_node("old_type", "old_text")

        record = strategy._substitute(node, "new_type", "new_text", mock_rule)

        assert node.type == "new_type"
        assert node.text == "new_text"
        mock_rule.record_substitute.assert_called_once_with(node, "old_type")
        assert record == mock_rule.record_substitute.return_value


class TestBaseStrategyIndentation:
    """Tests for the indentation detection helper."""

    def test_get_indent_success(self, strategy):
        """It should return the text of the preceding whitespace node."""
        parent = make_test_node("suite")
        ws = make_test_node("whitespace", "    ")
        target = make_test_node("for", "for")

        parent.children = [ws, target]
        target.parent = parent

        assert strategy._get_indent(target) == "    "

    def test_get_indent_no_parent(self, strategy):
        """It should return empty string if the node has no parent."""
        node = make_test_node()
        assert strategy._get_indent(node) == ""

    def test_get_indent_first_child(self, strategy):
        """It should return empty string if the node is the first child of its parent."""
        parent = make_test_node("module")
        target = make_test_node("for")
        parent.children = [target]
        target.parent = parent

        assert strategy._get_indent(target) == ""

    def test_get_indent_preceded_by_non_whitespace(self, strategy):
        """It should return empty string if the preceding sibling is not whitespace."""
        parent = make_test_node("block")
        prev = make_test_node("comment", "# note")
        target = make_test_node("for")

        parent.children = [prev, target]
        target.parent = parent

        assert strategy._get_indent(target) == ""
