"""Tests for the destructed-name generator used by identifier renaming.
"""

from unittest.mock import Mock

from transtructiver.mutation.rules.identifier_renaming._rename_destruction import (
    _build_destructed_name,
)
from transtructiver.node import Node


def test_build_destructed_name_maps_common_types():
    """Verify common context types map to the expected compact codes."""
    mapping = {
        "list": "l",
        "arr": "a",
        "dict": "d",
        "str": "s",
        "int": "i",
        "num": "n",
        "bool": "b",
        "func": "f",
        "df": "d",
        "conn": "c",
    }

    for ctx, code in mapping.items():
        node = Node((0, 0), (0, 5), "identifier", text=f"orig_{ctx}")
        result = _build_destructed_name(node, "python", None)
        assert result == code


def test_build_destructed_name_fallback_for_unknown_context():
    """Return 'x' when context_type contains no known keys."""
    node = Node((0, 0), (0, 5), "identifier", text="orig")

    result = _build_destructed_name(node, "python", None)
    assert result != "orig"
    assert result.isalpha()
    assert len(result) == 1


def test_empty_node_text_returns_empty_string():
    """Return empty string when the node has no text."""
    node = Node((0, 0), (0, 0), "identifier", text="")

    result = _build_destructed_name(node, "python", None)
    assert result == ""


def test_formatting_applied_for_language(monkeypatch):
    """Ensure format_identifier is invoked so callers receive language-aware names."""
    mock_formatter = Mock(return_value="FORMATTED")
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_destruction.format_identifier",
        mock_formatter,
    )

    node = Node((0, 0), (0, 5), "identifier", text="orig_list")

    result = _build_destructed_name(node, "java", None)
    mock_formatter.assert_called_once()
    assert result == "FORMATTED"


def test_build_destructed_name_title():
    """Ensure formatting produces uppercase char when title."""
    node = Node((0, 0), (0, 5), "identifier", text="orig_list")
    node.semantic_label = "class_name"

    result = _build_destructed_name(node, "java", None)
    assert result == "L"
