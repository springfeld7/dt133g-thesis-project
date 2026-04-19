"""Tests for identifier renaming substitution strategy."""

import json
from unittest.mock import Mock

import pytest

from transtructiver.mutation.rules.identifier_renaming._rename_substitution import (
    _build_substitute_name,
)
from transtructiver.node import Node


@pytest.fixture
def mock_frequency_map(tmp_path):
    """Create a mock frequency map JSON file for testing."""
    freq_map = {
        "version": 1,
        "role_maps": {
            "python": {
                "variable_name": {
                    "number": {"count": 5, "index": 3, "total": 2},
                    "string": {"name": 4, "value": 2},
                },
                "parameter_name": {
                    "number": {"param": 6, "arg": 4},
                    "none": {"x": 2, "y": 1},
                },
                "function_name": {
                    "none": {"process": 5, "execute": 3, "run": 2},
                },
            },
            "java": {
                "variable_name": {
                    "number": {"counter": 10, "sum": 5},
                    "none": {"obj": 3},
                },
            },
        },
    }

    freq_file = tmp_path / "frequency_map.json"
    with open(freq_file, "w") as f:
        json.dump(freq_map, f)

    return str(freq_file)


def test_substitute_with_matching_context_type(mock_frequency_map, monkeypatch):
    """Test substitution when context_type matches frequency map exactly."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {
            "number": {"count": 5, "index": 3},
            "string": {"name": 4},
        },
    )

    node = Node((0, 0), (0, 5), "identifier", text="original")
    node.semantic_label = "variable_name"
    node.context_type = "number"

    result = _build_substitute_name(node, "python")

    # Result should be one of the identifiers from the "number" context
    assert result in ["count", "index"]


def test_substitute_fallback_to_none_context(monkeypatch):
    """Test fallback to 'none' context when context_type has <20 matches."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {
            "number": {"count": 5},  # Only 1 identifier
            "none": {"fallback": 10, "backup": 8},
        },
    )

    node = Node((0, 0), (0, 5), "identifier", text="original")
    node.semantic_label = "variable_name"
    node.context_type = "number"

    result = _build_substitute_name(node, "python")

    # Should include fallback options
    assert result in ["count", "fallback", "backup"]


def test_substitute_context_type_none(monkeypatch):
    """Test substitution when context_type is 'none'."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {
            "none": {"process": 5, "execute": 3},
        },
    )

    node = Node((0, 0), (0, 5), "identifier", text="original")
    node.semantic_label = "function_name"
    node.context_type = "none"

    result = _build_substitute_name(node, "python")

    assert result in ["process", "execute"]


def test_substitute_no_frequency_map(monkeypatch):
    """Test that original text is returned when no frequency map is available."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {},
    )

    node = Node((0, 0), (0, 5), "identifier", text="original")
    node.semantic_label = "variable_name"
    node.context_type = "number"

    result = _build_substitute_name(node, "python")

    assert result == "original"


def test_substitute_empty_node_text():
    """Test handling of node with empty text."""
    node = Node((0, 0), (0, 0), "identifier", text="")
    node.semantic_label = "variable_name"
    node.context_type = "number"

    result = _build_substitute_name(node, "python")

    assert result == ""


def test_substitute_unsupported_language(monkeypatch):
    """Test that unsupported languages return original text."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"number": {"sub": 5}},
    )

    node = Node((0, 0), (0, 5), "identifier", text="original")
    node.semantic_label = "variable_name"
    node.context_type = "number"

    result = _build_substitute_name(node, "rust")  # Unsupported language

    assert result == "original"


def test_substitute_missing_semantic_label(monkeypatch):
    """Test handling of node without semantic_label."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"number": {"sub": 5}},
    )

    node = Node((0, 0), (0, 5), "identifier", text="original")
    node.semantic_label = None
    node.context_type = "number"

    result = _build_substitute_name(node, "python")

    # Should return original since no semantic label
    assert result == "original"


def test_substitute_multiple_languages(monkeypatch):
    """Test substitution with different languages."""

    def mock_load(path, lang, label):
        maps = {
            "python": {"number": {"py_var": 5}},
            "java": {"number": {"javaVar": 5}},
            "cpp": {"number": {"cpp_var": 5}},
        }
        return maps.get(lang, {})

    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        mock_load,
    )

    node = Node((0, 0), (0, 5), "identifier", text="original")
    node.semantic_label = "variable_name"
    node.context_type = "number"

    py_result = _build_substitute_name(node, "python")
    java_result = _build_substitute_name(node, "java")
    cpp_result = _build_substitute_name(node, "cpp")

    assert py_result == "py_var"
    assert java_result == "javaVar"
    assert cpp_result == "cppVar"


def test_substitute_deterministic_with_seed(monkeypatch):
    """Test that substitution is deterministic with fixed seed."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {
            "number": {f"id{i}": i for i in range(100)},
        },
    )

    node = Node((0, 0), (0, 5), "identifier", text="original")
    node.semantic_label = "variable_name"
    node.context_type = "number"

    # Same node and language should produce same result (due to fixed seed=42)
    result1 = _build_substitute_name(node, "python")
    result2 = _build_substitute_name(node, "python")

    assert result1 == result2


def test_substitute_formatting_applied(monkeypatch):
    """Test that format_identifier is called to apply formatting."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"number": {"snake_case": 5}},
    )

    # Mock format_identifier to track calls
    mock_formatter = Mock(return_value="FORMATTED_snake_case")
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.format_identifier",
        mock_formatter,
    )

    node = Node((0, 0), (0, 5), "identifier", text="original")
    node.semantic_label = "variable_name"
    node.context_type = "number"

    result = _build_substitute_name(node, "python")

    # Verify format_identifier was called
    mock_formatter.assert_called()
    assert result == "FORMATTED_snake_case"
