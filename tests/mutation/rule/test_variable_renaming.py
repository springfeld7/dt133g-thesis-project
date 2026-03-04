"""Unit tests for the VariableRenaming mutation rule."""

import pytest
from src.transtructiver.mutation.mutation_types import MutationAction
from src.transtructiver.mutation.rule.variable_renaming import VariableRenaming
from src.transtructiver.node import Node


@pytest.fixture
def sample_tree():
    """Create a sample node tree with identifiers for testing.

    Returns a root node with 4 children:
    - [0]: identifier "x"
    - [1]: identifier "y"
    - [2]: identifier "x"
    - [3]: number "1"
    """
    root = Node((0, 0), (0, 10), "module", children=[])
    first_x = Node((1, 0), (1, 1), "identifier", text="x")
    first_y = Node((1, 2), (1, 3), "identifier", text="y")
    second_x = Node((1, 4), (1, 5), "identifier", text="x")
    non_identifier = Node((1, 6), (1, 7), "number", text="1")
    root.children = [first_x, first_y, second_x, non_identifier]

    return root


def test_variable_renaming_initialization():
    """Ensure the subclass initializes base and custom fields correctly."""
    rule = VariableRenaming()

    assert rule.name == "VariableRenaming"


def test_variable_renaming_apply_with_none_root():
    """Ensure apply handles None root safely."""
    rule = VariableRenaming()

    assert rule.apply(None) == []


def test_variable_renaming_mutates_node_text(sample_tree):
    """Ensure apply renames identifier node text values consistently."""
    first_x, first_y, second_x, non_identifier = sample_tree.children

    rule = VariableRenaming()
    rule.apply(sample_tree)

    assert first_x.text != "x"
    assert first_y.text != "y"
    assert second_x.text != "x"

    assert non_identifier.text == "1"
    assert second_x.text == first_x.text


def test_variable_renaming_returns_mutation_records(sample_tree):
    """Ensure apply returns correct mutation records for renamed identifiers."""
    first_x, first_y, second_x, _ = sample_tree.children

    rule = VariableRenaming()
    records = rule.apply(sample_tree)

    assert len(records) == 3
    assert all(record.action == MutationAction.RENAME for record in records)
    assert records[0].node_id == (1, 0)

    assert records[0].metadata != {"new_val": "x"}
    assert records[0].metadata == {"new_val": first_x.text}

    assert records[1].metadata != {"new_val": "y"}
    assert records[1].metadata == {"new_val": first_y.text}

    assert records[2].metadata == {"new_val": second_x.text}
    assert records[2].metadata == {"new_val": first_x.text}
