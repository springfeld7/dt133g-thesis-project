"""
Unit tests for PythonInsertionStrategy.

Covers:
- Indentation prefix calculation
- Gap validation
- Terminal detection
- Integration of mixed child block scopes
"""

from platform import node

from numpy import block
import pytest
from src.transtructiver.mutation.rules.dead_code_insertion.insertion_strategies.python_strategy import (
    PythonInsertionStrategy,
)
from src.transtructiver.node import Node


@pytest.fixture
def strategy():
    """Provides a PythonInsertionStrategy instance."""
    return PythonInsertionStrategy()


# ===== Helper Node Factory =====
def make_node(
    type_: str,
    text: str = "",
    column: int = 0,
    children: list[Node] | None = None,
    parent: Node | None = None,
) -> Node:
    """Factory for Node instances with optional parent and children setup."""
    node = Node(
        start_point=(0, column), end_point=(0, column), type=type_, text=text, children=children
    )
    node.parent = parent
    if children:
        for child in children:
            child.parent = node
    return node


# ===== Test Class =====
class TestPythonInsertionStrategy:
    """Test class for PythonInsertionStrategy."""

    # --- Indentation Prefix ---

    def test_first_child_returns_none(self, strategy):
        """Returns None if node is the first child of its parent."""
        parent = make_node("block_scope")
        node = make_node("code", column=4, parent=parent)
        parent.children.append(node)  # node is first child
        assert strategy.get_indent_prefix(node) is None

    def test_first_child_returns_none(self, strategy):
        """Returns None if node is the first child of its parent."""
        parent = make_node("block_scope")
        node = make_node("code", column=4, parent=parent)
        parent.children.append(node)
        assert strategy.get_indent_prefix(node) is None

    def test_preceding_non_whitespace_uses_column(self, strategy):
        """Falls back to node column if preceding sibling is not whitespace."""
        code1 = make_node("expression_statement", column=2)
        node = make_node("code", column=4)
        parent = make_node("block_scope", children=[code1, node])
        code1.parent = parent
        node.parent = parent
        assert strategy.get_indent_prefix(node) == ""

    def test_zero_column(self, strategy):
        """Handles zero column correctly."""
        ws = make_node("whitespace", text="", column=0)
        node = make_node("code", column=0)
        parent = make_node("block_scope", children=[ws, node])
        ws.parent = parent
        node.parent = parent
        assert strategy.get_indent_prefix(node) == ""

    def test_large_column(self, strategy):
        """Handles large column values correctly."""
        ws = make_node("whitespace", text=" " * 20, column=20)
        node = make_node("code", column=20)
        parent = make_node("block_scope", children=[ws, node])
        ws.parent = parent
        node.parent = parent
        prefix = strategy.get_indent_prefix(node)
        assert len(prefix) == 20
        assert set(prefix) == {" "}

    # --- Valid Container Tests ---

    def test_valid_container_returns_true(self, strategy):
        """Returns True for a proper multi-line block_scope."""
        child1 = make_node("expression_statement")
        child2 = make_node("return_statement")
        block = make_node("block_scope", children=[child1, child2])
        for c in block.children:
            c.parent = block

        # Parent node
        parent = make_node("function_def")
        parent.start_point = (0, 0)  # row 0
        block.parent = parent
        block.start_point = (1, 4)  # different row from parent
        assert strategy.is_valid_container(block) is True

    def test_container_with_pass_returns_false(self, strategy):
        """Returns False if any child is a pass statement."""
        child1 = make_node("pass_statement")
        block = make_node("block_scope", children=[child1])
        for c in block.children:
            c.parent = block

        parent = make_node("function_def")
        parent.start_point = (0, 0)
        block.parent = parent
        block.start_point = (1, 4)  # different row from parent
        assert strategy.is_valid_container(block) is False

    def test_single_line_block_returns_false(self, strategy):
        """Returns False if block starts on same row as parent (single-line)."""

        child1 = make_node("expression_statement")
        block = make_node("block_scope", children=[child1])
        for c in block.children:
            c.parent = block

        parent = make_node("function_def")
        parent.start_point = (0, 0)
        block.parent = parent
        block.start_point = (0, 4)  # same row as parent
        assert strategy.is_valid_container(block) is False

    # --- Gap Validation ---

    def test_gap_at_block_start(self, strategy):
        """Gap is valid at block start (preceding is None)."""
        node = make_node("code")
        assert strategy.is_valid_gap(node, preceding=None) is True

    def test_gap_after_whitespace(self, strategy):
        """Gap is valid if preceded by a whitespace node."""
        ws = make_node("whitespace")
        node = make_node("code")
        assert strategy.is_valid_gap(node, preceding=ws) is True

    def test_gap_after_non_whitespace(self, strategy):
        """Gap is invalid if preceded by a non-whitespace node."""
        code = make_node("expression_statement")
        node = make_node("code")
        assert strategy.is_valid_gap(node, preceding=code) is False

    def test_gap_after_newline_is_invalid(self, strategy):
        """Gap after newline node is invalid."""
        nl = make_node("newline")
        node = make_node("code")
        assert strategy.is_valid_gap(node, preceding=nl) is False

    # --- Terminal Detection ---

    @pytest.mark.parametrize(
        "typ", ["return_statement", "break_statement", "continue_statement", "pass_statement"]
    )
    def test_terminal_statements(self, strategy, typ):
        """Correctly identifies terminal statements."""
        node = make_node(typ)
        assert strategy.is_terminal(node) is True

    @pytest.mark.parametrize(
        "typ", ["expression_statement", "raise_statement", "if_statement", "while_statement"]
    )
    def test_non_terminal_statements(self, strategy, typ):
        """Correctly identifies non-terminal statements."""
        node = make_node(typ)
        assert strategy.is_terminal(node) is False

    def test_terminal_with_type_none(self, strategy):
        """Returns False if node type is None."""
        node = make_node(None)
        assert strategy.is_terminal(node) is False

    # --- Integration: block_scope with mixed children ---

    def test_mixed_block_structure(self, strategy):
        """Handles block_scope with mixed whitespace and code children."""
        ws1 = make_node("whitespace", text="  ", column=2)
        code1 = make_node("expression_statement", column=2)
        ws2 = make_node("whitespace", text="    ", column=4)
        code2 = make_node("return_statement", column=4)
        block = make_node("block_scope", children=[ws1, code1, ws2, code2])
        for c in block.children:
            c.parent = block

        # Indent prefixes
        assert strategy.get_indent_prefix(code1) == "  "
        assert strategy.get_indent_prefix(code2) == "    "

        # Gap validity
        assert strategy.is_valid_gap(code1, ws1) is True
        assert strategy.is_valid_gap(code2, ws2) is True
        assert strategy.is_valid_gap(code2, code1) is False

        # Terminal detection
        assert strategy.is_terminal(code1) is False
        assert strategy.is_terminal(code2) is True
