"""Unit tests for mutation_types.py

Validates that MutationAction metadata contracts are correctly enforced.
Covers positive cases, structural violations, and edge cases.
"""

import pytest
from src.transtructiver.mutation.mutation_types import MutationAction, validate_action_metadata


# ===== Helpers =====


def valid_coord():
    """Return a valid coordinate tuple."""
    return (10, 5)


# ===== Positive Cases of Correct Metadata =====


def test_rename_valid_metadata():
    """
    Test that RENAME accepts valid metadata.
    """
    metadata = {"new_val": "renamed_var"}
    assert validate_action_metadata(MutationAction.RENAME, metadata)


def test_delete_valid_metadata():
    """
    Test that DELETE accepts valid metadata containing required keys.
    """
    metadata = {"node_type": "comment", "content": "// a comment"}
    assert validate_action_metadata(MutationAction.DELETE, metadata)


def test_insert_valid_metadata():
    """
    Test that INSERT accepts valid metadata.
    """
    metadata = {"new_val": "x = 1", "node_type": "Assign", "insertion_point": (0, 0)}
    assert validate_action_metadata(MutationAction.INSERT, metadata)


def test_substitute_valid_metadata():
    """
    Test that SUBSTITUTE accepts properly structured parts_map.
    """
    metadata = {
        "old_type": "for_statement",
        "new_type": "while_statement",
        "new_val": "while (condition) { body }",
    }

    assert validate_action_metadata(MutationAction.SUBSTITUTE, metadata)


def test_flatten_valid_metadata():
    """
    Test that FLATTEN accepts valid ref_map.
    """
    metadata = {
        "node_type": "If",
        "ref_map": {
            "entry": valid_coord(),
            "exit": valid_coord(),
        },
    }
    assert validate_action_metadata(MutationAction.FLATTEN, metadata)


# ===== Missing Required Keys =====


@pytest.mark.parametrize(
    "action,metadata",
    [
        (MutationAction.RENAME, {}),
        (MutationAction.INSERT, {"new_val": "x"}),
        (MutationAction.DELETE, {}),
        (MutationAction.SUBSTITUTE, {"node_type": "For"}),
    ],
)
def test_missing_required_keys(action, metadata):
    """
    Test that missing required metadata keys raise ValueError.
    """
    with pytest.raises(ValueError):
        validate_action_metadata(action, metadata)


# ==== Invalid Map Type =====


def test_flatten_ref_map_not_dict():
    """
    Test that FLATTEN fails if ref_map is not a dictionary.
    """
    metadata = {
        "node_type": "If",
        "ref_map": 123,
    }

    with pytest.raises(TypeError):
        validate_action_metadata(MutationAction.FLATTEN, metadata)


# ==== Extra Metadata Keys (Allowed Behavior) ====


def test_extra_top_level_keys_allowed():
    """
    Test that extra top-level metadata keys do not break validation.
    """
    metadata = {
        "new_val": "x",
        "extra": "ignored",
    }

    assert validate_action_metadata(MutationAction.RENAME, metadata)


# ==== MutationAction Properties ====


@pytest.mark.parametrize(
    "action,expected",
    [
        (MutationAction.INSERT, True),
        (MutationAction.DELETE, True),
        (MutationAction.FLATTEN, True),
        (MutationAction.SUBSTITUTE, True),
        (MutationAction.RENAME, False),
        (MutationAction.REFORMAT, False),
    ],
)
def test_is_structural_property(action, expected):
    """
    Ensure that the is_structural property returns True for structural actions
    and False for non-structural actions.
    """
    assert action.is_structural == expected
