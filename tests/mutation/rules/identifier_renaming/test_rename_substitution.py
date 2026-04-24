"""Unit tests for identifier renaming substitution logic with similarity thresholds."""

from unittest.mock import Mock
from transtructiver.mutation.rules.identifier_renaming._rename_substitution import (
    _build_substitute_name,
)
from transtructiver.node import Node


def test_returns_candidate_within_similarity(monkeypatch):
    """Should return candidate if similarity is within thresholds."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"none": {"fob": 1}},
    )
    node = Node((0, 0), (0, 5), "identifier", text="foo")
    node.semantic_label = "variable_name"
    node.context_type = "none"
    assert _build_substitute_name(node, "python") == "fob"


def test_rejects_too_similar_candidate(monkeypatch):
    """Should return original if all candidates is too similar (ratio=1.0)."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"none": {"foo": 1}},
    )
    node = Node((0, 0), (0, 5), "identifier", text="foo")
    node.semantic_label = "variable_name"
    node.context_type = "none"
    assert _build_substitute_name(node, "python") == "foo"


def test_fallback_to_dissimilar_candidate(monkeypatch):
    """Should return dissimilar if similar candidate is missing."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"none": {"bar": 1}},
    )
    node = Node((0, 0), (0, 5), "identifier", text="foo")
    node.semantic_label = "variable_name"
    node.context_type = "none"
    assert _build_substitute_name(node, "python") == "bar"


def test_multiple_candidates_selects_valid(monkeypatch):
    """Should select a valid candidate from multiple options."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"none": {"foo": 1, "fob": 1, "bar": 1}},
    )
    node = Node((0, 0), (0, 5), "identifier", text="foo")
    node.semantic_label = "variable_name"
    node.context_type = "none"
    assert _build_substitute_name(node, "python") == "fob"


def test_context_type_and_fallback(monkeypatch):
    """Should use fallback context if needed and select valid candidate."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {
            "number": {},
            "none": {"fallback": 10, "fub": 8},
        },
    )
    node = Node((0, 0), (0, 5), "identifier", text="foo")
    node.semantic_label = "variable_name"
    node.context_type = "number"
    assert _build_substitute_name(node, "python") == "fub"


def test_returns_original_if_no_valid_candidates(monkeypatch):
    """Should return original if no candidates are within similarity thresholds."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"none": {"completelydifferent": 1}},
    )
    node = Node((0, 0), (0, 5), "identifier", text="foo")
    node.semantic_label = "variable_name"
    node.context_type = "none"
    assert _build_substitute_name(node, "python") == "foo"


def test_empty_node_text_returns_empty():
    """Should return empty string if node.text is empty."""
    node = Node((0, 0), (0, 0), "identifier", text="")
    node.semantic_label = "variable_name"
    node.context_type = "number"
    assert _build_substitute_name(node, "python") == ""


def test_no_frequency_map_returns_original(monkeypatch):
    """Should return original if frequency map is empty."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {},
    )
    node = Node((0, 0), (0, 5), "identifier", text="foo")
    node.semantic_label = "variable_name"
    node.context_type = "number"
    assert _build_substitute_name(node, "python") == "foo"


def test_unsupported_language_returns_original(monkeypatch):
    """Should return original if language is unsupported."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"number": {"sub": 5}},
    )
    node = Node((0, 0), (0, 5), "identifier", text="foo")
    node.semantic_label = "variable_name"
    node.context_type = "number"
    assert _build_substitute_name(node, "rust") == "foo"


def test_missing_semantic_label_returns_original(monkeypatch):
    """Should return original if semantic_label is missing."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"number": {"sub": 5}},
    )
    node = Node((0, 0), (0, 5), "identifier", text="foo")
    node.semantic_label = None
    node.context_type = "number"
    assert _build_substitute_name(node, "python") == "foo"


def test_multiple_languages(monkeypatch):
    """Should select correct candidate for each supported language."""

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

    node_py = Node((0, 0), (0, 5), "identifier", text="pyth_v")
    node_py.semantic_label = "variable_name"
    node_py.context_type = "number"
    node_java = Node((0, 0), (0, 5), "identifier", text="javVar")
    node_java.semantic_label = "variable_name"
    node_java.context_type = "number"
    node_cpp = Node((0, 0), (0, 5), "identifier", text="cpl_va")
    node_cpp.semantic_label = "variable_name"
    node_cpp.context_type = "number"
    assert _build_substitute_name(node_py, "python") == "py_var"
    assert _build_substitute_name(node_java, "java") == "javaVar"
    assert _build_substitute_name(node_cpp, "cpp") == "cppVar"


def test_deterministic_with_seed(monkeypatch):
    """Should always return the same result for the same input (deterministic)."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {
            "number": {f"id{i}": i for i in range(100)},
        },
    )
    node = Node((0, 0), (0, 5), "identifier", text="foo")
    node.semantic_label = "variable_name"
    node.context_type = "number"
    result1 = _build_substitute_name(node, "python")
    result2 = _build_substitute_name(node, "python")
    assert result1 == result2


def test_formatting_applied(monkeypatch):
    """Should call format_identifier on the chosen candidate."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.load_identifier_frequency_map",
        lambda path, lang, label: {"number": {"snake_case": 5}},
    )
    mock_formatter = Mock(return_value="FORMATTED_snake_case")
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.format_identifier",
        mock_formatter,
    )
    # Use an old_text with a single character difference to "snake_case" to pass similarity and length checks
    node = Node((0, 0), (0, 5), "identifier", text="snake_styled")
    node.semantic_label = "variable_name"
    node.context_type = "number"
    result = _build_substitute_name(node, "python")
    mock_formatter.assert_called()
    assert result == "FORMATTED_snake_case"
