"""Unit tests for the MutationRule abstract base class and MutationRecord schema.

This module verifies:
- The abstract nature of the MutationRule class (cannot be instantiated).
- Correct initialization of rule metadata (e.g., automatic name assignment).
- Compliance of concrete implementations with the MutationRecord schema.
- Proper use of the MutationAction enum for type-safe reporting.
- Support for both original source nodes and synthetic/inserted nodes.

Testing is performed using minimal concrete subclasses to validate the 
interface contract before integration into the MutationEngine.
"""

import pytest
from typing import List, cast
from transtructiver.mutation.rules.mutation_rule import MutationRule, MutationRecord
from transtructiver.mutation.mutation_types import MutationAction
from transtructiver.mutation.mutation_context import MutationContext
from transtructiver.node import Node


def test_cannot_instantiate_abc():
    """Ensure that the MutationRule ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        MutationRule()  # type: ignore[abstract]


def test_concrete_rule_implementation():
    """Test a minimal valid implementation of a MutationRule."""

    class RenameVariableRule(MutationRule):
        def apply(self, root, context) -> List[MutationRecord]:
            # Now returning an actual object, not a dict
            return [
                MutationRecord(
                    node_id=(10, 5),
                    action=MutationAction.RENAME,
                    metadata={"old_val": "x", "new_val": "x_var"},
                )
            ]

    rule = RenameVariableRule()

    rule = RenameVariableRule()
    assert rule.name == "RenameVariableRule"

    node = Node((0, 0), (0, 0), "")
    results = rule.apply(node, MutationContext())
    assert isinstance(results[0], MutationRecord)
    assert results[0].node_id == (10, 5)
    assert results[0].action == MutationAction.RENAME


def test_synthetic_node_record():
    """Verify synthetic nodes can use negative coordinates"""

    class InsertDeadCodeRule(MutationRule):
        def apply(self, root, context) -> List[MutationRecord]:
            return [
                MutationRecord(
                    node_id=(-1, -1),
                    action=MutationAction.INSERT,
                    metadata={
                        "new_val": "pass",
                        "node_type": "stmt",
                        "insertion_point": (0, 0),  # required by verifier
                    },
                )
            ]

    rule = InsertDeadCodeRule()
    node = Node((0, 0), (0, 0), "")
    results = rule.apply(node, MutationContext())

    # Check synthetic coordinates
    assert results[0].node_id[0] < 0
    # Check action
    assert results[0].action == MutationAction.INSERT
    # Check insertion_point exists
    assert "insertion_point" in results[0].metadata
    assert results[0].metadata["insertion_point"] == (0, 0)


def test_invalid_metadata_raises_error():
    """Verify that malformed records blow up immediately (the new behavior)."""
    with pytest.raises(ValueError, match="missing required key"):
        MutationRecord(
            node_id=(1, 1), action=MutationAction.RENAME, metadata={}  # Missing 'new_val'
        )


def test_repr_output():
    """Verify the string representation for debugging."""

    class TestRule(MutationRule):
        def apply(self, root, context):
            return []

    rule = TestRule()
    assert repr(rule) == "<TestRule>"


def test_record_substitute_creates_valid_record():
    """Ensure SUBSTITUTE record is created with correct metadata."""

    class DummyNode:
        def __init__(self):
            self.start_point = (3, 7)
            self.type = "new_type"
            self.text = "new_text"

    class TestRule(MutationRule):
        def apply(self, root, context):
            return []

    node = DummyNode()
    rule = TestRule()

    record = rule.record_substitute(cast(Node, node), old_type="old_type")

    assert record.node_id == (3, 7)
    assert record.action == MutationAction.SUBSTITUTE
    assert record.metadata == {
        "old_type": "old_type",
        "new_type": "new_type",
        "new_val": "new_text",
    }
