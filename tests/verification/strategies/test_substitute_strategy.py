"""Unit tests for the SubstituteVerificationStrategy.

Validates that the SubstituteVerificationStrategy correctly audits structural 
substitutions by checking type transitions, metadata consistency, and content.
"""

import pytest
from typing import Optional
from src.transtructiver.verification.strategies.substitute_strategy import (
    SubstituteVerificationStrategy,
)
from src.transtructiver.mutation.mutation_manifest import ManifestEntry
from src.transtructiver.node import Node


# ===== Helpers =====


def make_node(
    type: str = "for",
    text: str = "for",
    start: tuple[int, int] = (0, 0),
    end: tuple[int, int] = (0, 3),
) -> Node:
    """Helper to construct a Node for testing."""
    return Node(start_point=start, end_point=end, type=type, text=text)


def make_manifest_entry(
    old_type: str, new_type: str, new_val: str, original_id: tuple[int, int] = (0, 0)
) -> ManifestEntry:
    """Helper to construct a ManifestEntry with substitution metadata."""
    return ManifestEntry(
        original_id=original_id,
        history=[],
        metadata={"old_type": old_type, "new_type": new_type, "new_val": new_val},
    )


# ===== Positive Cases =====


def test_valid_substitution():
    """It should return an empty list when type transitions and content match perfectly."""
    orig = make_node(type="for", text="for")
    mut = make_node(type="while", text="while")
    entry = make_manifest_entry(old_type="for", new_type="while", new_val="while")

    strategy = SubstituteVerificationStrategy()
    errors = strategy.verify(orig, mut, entry)
    assert errors == []


def test_valid_complex_statement_substitution():
    """It should validate a full statement transition (e.g., for_statement to while_statement)."""
    orig = make_node(type="for_statement", text="for x in y: pass")
    mut = make_node(type="while_statement", text="while True: pass")
    entry = make_manifest_entry(
        old_type="for_statement", new_type="while_statement", new_val="while True: pass"
    )

    strategy = SubstituteVerificationStrategy()
    errors = strategy.verify(orig, mut, entry)
    assert errors == []


# ===== Existence Checks =====


@pytest.mark.parametrize(
    "orig, mut, expected_msg",
    [
        (None, make_node(), "Original node missing"),
        (make_node(), None, "Mutated node missing"),
    ],
)
def test_missing_nodes(orig: Optional[Node], mut: Optional[Node], expected_msg: str):
    """It should return an error if either the original or mutated node is None."""
    entry = make_manifest_entry("for", "while", "while")
    strategy = SubstituteVerificationStrategy()
    errors = strategy.verify(orig, mut, entry)

    assert len(errors) == 1
    assert expected_msg in errors[0]


# ===== Metadata & Type Mismatches =====


def test_manifest_mismatch_with_original():
    """It should fail if the manifest's 'old_type' doesn't match the actual original node type."""
    orig = make_node(type="enhanced_for_statement")
    mut = make_node(type="while_statement")
    # Manifest claims the old type was a simple 'for'
    entry = make_manifest_entry(old_type="for", new_type="while_statement", new_val="while")

    strategy = SubstituteVerificationStrategy()
    errors = strategy.verify(orig, mut, entry)

    assert any("Manifest Mismatch" in e for e in errors)


def test_type_mismatch_mutated():
    """It should fail if the mutated node's type does not match the 'new_type' in metadata."""
    orig = make_node(type="for")
    mut = make_node(type="identifier", text="while")  # Wrong type
    entry = make_manifest_entry(old_type="for", new_type="while", new_val="while")

    strategy = SubstituteVerificationStrategy()
    errors = strategy.verify(orig, mut, entry)

    assert any("Type Mismatch" in e for e in errors)


def test_content_mismatch_mutated():
    """It should fail if the text content of the mutated node differs from 'new_val'."""
    orig = make_node(type="for", text="for")
    mut = make_node(type="while", text="while_wrong")
    entry = make_manifest_entry(old_type="for", new_type="while", new_val="while")

    strategy = SubstituteVerificationStrategy()
    errors = strategy.verify(orig, mut, entry)

    assert any("Content Mismatch" in e for e in errors)


# ===== Logic Transition Validation =====


def test_invalid_logic_transition():
    """It should fail if the substitution pair is not in the recognized whitelist."""
    orig = make_node(type="for", text="for")
    mut = make_node(type="if_statement", text="if True:")  # Invalid transition
    entry = make_manifest_entry(old_type="for", new_type="if_statement", new_val="if True:")

    strategy = SubstituteVerificationStrategy()
    errors = strategy.verify(orig, mut, entry)

    assert any("Invalid Logic Transition" in e for e in errors)


@pytest.mark.parametrize(
    "old_t, new_t, expected",
    [
        ("for", "while", True),
        ("in", "true", True),
        ("enhanced_for_statement", "while_statement", True),
        ("identifier", "while", False),
    ],
)
def test_is_valid_transformation_internal(old_t: str, new_t: str, expected: bool):
    """Internal check for the transformation whitelist logic."""
    strategy = SubstituteVerificationStrategy()
    assert strategy._is_valid_transformation(old_t, new_t) == expected
