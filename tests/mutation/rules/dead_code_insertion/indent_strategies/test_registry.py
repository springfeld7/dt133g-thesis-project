"""Unit tests for the indentation strategy registry."""

from src.transtructiver.mutation.rules.dead_code_insertion.indent_strategies.registry import (
    get_indentation_prefix,
)
from src.transtructiver.mutation.rules.dead_code_insertion.indent_strategies.registry import (
    _STRATEGY_MAP,
)
from src.transtructiver.mutation.rules.dead_code_insertion.indent_strategies.python_strategy import (
    PythonIndent,
)
from src.transtructiver.mutation.rules.dead_code_insertion.indent_strategies.cstyle_strategy import (
    CStyleIndent,
)


# ===== Dummy Node Classes =====


class DummyPythonNode:
    """Python block node with start_point (line, column)."""

    def __init__(self, column: int):
        self.start_point = (0, column)


class DummyCNode:
    """C-style block node with children."""

    def __init__(self, children):
        self.children = children


class DummyChild:
    """Child node with type and text."""

    def __init__(self, type_: str, text: str):
        self.type = type_
        self.text = text


# ===== Test Class =====


class TestIndentStrategyRegistry:

    def test_python_strategy_returns_correct_prefix(self):
        """PythonIndent returns spaces matching start_point[1]."""
        node = DummyPythonNode(column=4)
        assert get_indentation_prefix(node, "python") == "    "

    def test_python_strategy_none_for_missing_column(self):
        """PythonIndent returns None if start_point[1] is None."""

        class Node:
            start_point = (0, None)

        node = Node()
        assert get_indentation_prefix(node, "python") is None

    def test_java_strategy_returns_first_whitespace(self):
        """CStyleIndent returns first whitespace child for Java nodes."""
        children = [
            DummyChild("code", "int x;"),
            DummyChild("whitespace", "  "),
            DummyChild("newline", "\n"),
            DummyChild("code", "y=0;"),
        ]
        node = DummyCNode(children)
        assert get_indentation_prefix(node, "java") == "  "

    def test_cpp_strategy_returns_first_whitespace(self):
        """CStyleIndent returns first whitespace child for C++ nodes."""
        children = [
            DummyChild("whitespace", "    "),
            DummyChild("newline", "\n"),
            DummyChild("code", "int y;"),
        ]
        node = DummyCNode(children)
        assert get_indentation_prefix(node, "cpp") == "    "

    def test_cstyle_strategy_none_if_no_whitespace(self):
        """CStyleIndent returns None when no whitespace child exists."""
        children = [DummyChild("code", "x=1;")]
        node = DummyCNode(children)
        assert get_indentation_prefix(node, "java") is None
        assert get_indentation_prefix(node, "cpp") is None

    def test_unknown_language_returns_none(self):
        """Unknown language identifier returns None."""
        node = DummyPythonNode(column=2)
        assert get_indentation_prefix(node, "ruby") is None

    def test_registry_contains_expected_languages(self):
        """Registry maps language names to correct strategy types."""
        assert isinstance(_STRATEGY_MAP["python"], PythonIndent)
        assert isinstance(_STRATEGY_MAP["java"], CStyleIndent)
        assert isinstance(_STRATEGY_MAP["cpp"], CStyleIndent)

    def test_registry_language_case_insensitivity(self):
        """Language lookup in registry is case-insensitive."""
        node = DummyPythonNode(column=3)
        assert get_indentation_prefix(node, "Python") == "   "
        assert get_indentation_prefix(node, "PYTHON") == "   "

    def test_repeated_calls_are_deterministic(self):
        """Repeated calls for the same node return identical prefixes."""
        node = DummyPythonNode(column=5)
        prefix1 = get_indentation_prefix(node, "python")
        prefix2 = get_indentation_prefix(node, "python")
        assert prefix1 == prefix2
