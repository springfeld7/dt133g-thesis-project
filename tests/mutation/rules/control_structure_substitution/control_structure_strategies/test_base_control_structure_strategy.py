"""Unit tests for the BaseControlStructureStrategy.

Validates the concrete helper methods used for node insertion, 
substitution, and indentation detection in transformation strategies.
"""

import pytest
from unittest.mock import MagicMock
from transtructiver.node import Node
from transtructiver.mutation.mutation_context import MutationContext
from transtructiver.mutation.rules.control_structure_substitution.control_structure_strategies.base_control_structure_strategy import (
    BaseControlStructureStrategy,
)


# ===== Mock Implementation =====


class ConcreteStrategy(BaseControlStructureStrategy):
    """Concrete implementation for testing abstract base helpers."""

    def is_valid(self, node: Node) -> bool:
        return True

    def apply(self, node: Node, context: MutationContext, indent_unit: str):
        return []


# ===== Helpers =====


def make_test_node(node_type="test", text="text", start=(1, 0), end=(1, 4)):
    """Creates a basic node for testing."""
    return Node(start_point=start, end_point=end, type=node_type, text=text)


# ===== Fixtures =====


@pytest.fixture
def strategy():
    """Returns a concrete instance of the base strategy."""
    return ConcreteStrategy()


@pytest.fixture
def context():
    """Returns a mock MutationContext with incrementing IDs."""
    ctx = MagicMock(spec=MutationContext)
    ctx._counter = 100

    def next_id():
        ctx._counter += 1
        return ctx._counter

    ctx.next_id.side_effect = next_id
    return ctx


@pytest.fixture
def mock_rule():
    """Returns a mock MutationRule for recording changes."""
    return MagicMock()


# ===== Test Cases =====


class TestNodeInsertionHelpers:
    """Tests for _insert_node and _insert_segments methods."""

    def test_insert_node_logic(self, strategy, context, mock_rule):
        """It should create a node, link it to the parent, and record the insertion."""
        parent = make_test_node("block", "{ }")
        parent.children = []
        insertion_point = (1, 1)

        record = strategy._insert_node(
            context=context,
            parent=parent,
            point=insertion_point,
            type="statement",
            text="pass",
            index=0,
            rule=mock_rule,
        )

        # Structural Assertions
        assert len(parent.children) == 1
        new_node = parent.children[0]
        assert new_node.parent == parent
        assert new_node.text == "pass"
        assert new_node.type == "statement"
        assert new_node.start_point == (101, -1)

        # Record Assertions
        mock_rule.record_insert.assert_called_once_with(
            point=(101, -1), insertion_point=insertion_point, new_text="pass", new_type="statement"
        )
        assert record == mock_rule.record_insert.return_value

    def test_insert_segments_batch(self, strategy, context, mock_rule):
        """It should insert multiple segments and return a list of records."""
        parent = make_test_node("block", "{ }")
        parent.children = []
        segments = [("type1", "text1"), ("type2", "text2")]

        records = strategy._insert_segments(
            context=context, parent=parent, point=(2, 0), segments=segments, index=0, rule=mock_rule
        )

        assert len(parent.children) == 2
        assert len(records) == 2

        # Note: If index doesn't increment in the loop, the last item becomes first
        assert parent.children[0].text == "text2"
        assert parent.children[1].text == "text1"


class TestSubstitutionHelpers:
    """Tests for the _substitute method."""

    def test_substitute_updates_node_and_records(self, strategy, mock_rule):
        """It should update the node's attributes and call rule.record_substitute."""
        node = make_test_node("old_type", "old_text")

        record = strategy._substitute(node, "new_type", "new_text", mock_rule)

        assert node.type == "new_type"
        assert node.text == "new_text"
        mock_rule.record_substitute.assert_called_once_with(node, "old_type")
        assert record == mock_rule.record_substitute.return_value


class TestIndentationDetection:
    """Tests for the _get_indent method."""

    def test_get_indent_with_whitespace_sibling(self, strategy):
        """It should return the text of the immediately preceding whitespace node."""
        indent_node = make_test_node("whitespace", "    ")
        target_node = make_test_node("for", "for")
        parent = make_test_node("block")

        parent.children = [indent_node, target_node]
        target_node.parent = parent

        assert strategy._get_indent(target_node) == "    "

    def test_get_indent_no_parent(self, strategy):
        """It should return empty string if the node has no parent."""
        node = make_test_node()
        assert strategy._get_indent(node) == ""

    def test_get_indent_at_start_of_children(self, strategy):
        """It should return empty string if the node is the first child."""
        target_node = make_test_node()
        parent = make_test_node("block")
        parent.children = [target_node]
        target_node.parent = parent

        assert strategy._get_indent(target_node) == ""

    def test_get_indent_prev_sibling_not_whitespace(self, strategy):
        """It should return empty string if the previous sibling is not whitespace."""
        prev_node = make_test_node("comment", "# comment")
        target_node = make_test_node("for", "for")
        parent = make_test_node("block")

        parent.children = [prev_node, target_node]
        target_node.parent = parent

        assert strategy._get_indent(target_node) == ""
