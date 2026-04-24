"""Unit tests for CppForLoopStrategy in control structure substitution mutation rules.

Focuses on validating C++ specific node extraction (declaration, compound_statement)
and the transformation logic for converting for-loops to while-loops.
"""

import pytest
from unittest.mock import MagicMock, patch

from transtructiver.node import Node
from transtructiver.mutation.mutation_context import MutationContext
from transtructiver.mutation.rules.mutation_rule import MutationRule
from transtructiver.mutation.rules.control_structure_substitution.control_structure_strategies.for_loop_strategies.cpp_strategy import (
    CppForLoopStrategy,
)


class TestCppForLoopStrategy:
    """
    Comprehensive test suite for CppForLoopStrategy.
    Mirroring the Java recipe while adapting for C++ grammar (compound_statement).
    """

    @pytest.fixture
    def strategy(self):
        return CppForLoopStrategy()

    @pytest.fixture
    def mock_rule(self):
        rule = MagicMock(spec=MutationRule)
        rule.record_insert.side_effect = lambda **kwargs: {"type": "insert", **kwargs}
        rule.record_substitute.side_effect = lambda node, old_type: {
            "type": "substitute",
            "old": old_type,
        }
        rule.record_delete.side_effect = lambda parent, node: {"type": "delete", "node": node}
        rule.record_reformat.side_effect = lambda node, text: {"type": "reformat", "text": text}
        return rule

    @pytest.fixture
    def mock_context(self):
        ctx = MagicMock(spec=MutationContext)
        ctx._id_gen = 2000

        def next_id():
            ctx._id_gen += 1
            return ctx._id_gen

        ctx.next_id.side_effect = next_id
        return ctx

    def _n(self, n_type, text="", children=None):
        """Standard node factory to ensure parent-linking is handled automatically."""
        node = Node((0, 0), (0, len(text)), n_type, text)
        if children:
            node.children = children
            for child in children:
                child.parent = node
        return node

    # -------------------------------------------------------------------------
    # BASE LOGIC & C-STYLE UTILITIES
    # -------------------------------------------------------------------------

    def test_base_node_insertion(self, strategy, mock_context, mock_rule):
        parent = self._n("parent")
        record = strategy._insert_node(
            mock_context, parent, (5, 5), "test_type", "test_text", 0, mock_rule
        )

        assert len(parent.children) == 1
        assert parent.children[0].type == "test_type"
        assert record["new_type"] == "test_type"

    def test_cstyle_header_cleaning(self, strategy, mock_rule):
        semi = self._n(";", ";")
        ws = self._n("whitespace", " ")
        loop = self._n(
            "for_statement", children=[self._n("for"), self._n("("), semi, ws, self._n(")")]
        )

        records = strategy._clean_for_loop_header(loop, mock_rule)
        assert len(records) == 2
        mock_rule.record_delete.assert_any_call(loop, semi)

    # -------------------------------------------------------------------------
    # C++ SPECIFIC TESTS
    # -------------------------------------------------------------------------

    def test_extract_cpp_components(self, strategy):
        """Verifies C++ specific 'declaration' and 'compound_statement' extraction."""
        # Note: In C++ trees, the ';' is often a child of the declaration
        # OR handled as a section break. Your strategy expects 'declaration'
        # to consume the section.
        init = self._n("declaration", "int i=0;")  # Added ; to text for realism
        cond = self._n("binary_expression", "i<10")
        upd = self._n("update_expression", "i++")
        body = self._n("compound_statement", "{ f(); }")

        root = self._n(
            "for_statement",
            children=[
                self._n("for"),
                self._n("("),
                init,
                # REMOVED the extra self._n(";") here
                cond,
                self._n(";"),  # This is the second semicolon (before the update)
                upd,
                self._n(")"),
                body,
            ],
        )

        f, i, c, u, b = strategy._extract_for_loop_components(root)
        assert i == init
        assert c == cond
        assert u == upd
        assert b == body

    def test_apply_transformation_with_scoping(self, strategy, mock_rule, mock_context):
        """Integration: for(int i=0; i<1; i++) { body; } -> scoped while loop."""
        for_kw = self._n("for", "for")
        init = self._n("declaration", "int i=0")
        cond = self._n("binary_expression", "i<1")
        upd = self._n("update_expression", "i++")

        b_open, b_stmt, b_close = self._n("{"), self._n("body;"), self._n("}")
        body = self._n("compound_statement", children=[b_open, b_stmt, b_close])

        root = self._n(
            "for_statement",
            children=[
                for_kw,
                self._n("("),
                init,
                self._n(";"),
                cond,
                self._n(";"),
                upd,
                self._n(")"),
                body,
            ],
        )
        root.parent = self._n("module", children=[root])

        with patch.object(strategy, "_apply_indent_reformat", return_value=[]):
            records = strategy.apply(root, mock_rule, mock_context, "    ")

            # Check keyword substitution
            mock_rule.record_substitute.assert_called_with(for_kw, "for")
            assert for_kw.type == "while"

            # Check scoping braces insertion (1 for '{', 1 for '}')
            assert any(rec.get("new_type") == "{" for rec in records)
            assert any(rec.get("new_type") == "}" for rec in records)

    def test_infinite_loop_empty_condition(self, strategy, mock_rule, mock_context):
        """Verifies for(;;) results in 'true'."""
        close_paren = self._n(")", ")")
        root = self._n(
            "for_statement",
            children=[
                self._n("for"),
                self._n("("),
                self._n(";"),
                self._n(";"),
                close_paren,
                self._n("compound_statement", children=[self._n("{"), self._n("}")]),
            ],
        )

        strategy._insert_default_true_condition(root, mock_context, mock_rule)

        mock_rule.record_insert.assert_called_once_with(
            point=(2001, -1),
            insertion_point=close_paren.start_point,
            new_text="true",
            new_type="binary_expression",
        )

    def test_bailout_on_missing_braces(self, strategy, mock_rule, mock_context):
        """C++ allows for(...) stmt; - skip if no compound_statement braces exist."""
        root = self._n(
            "for_statement", children=[self._n("for"), self._n("expression_statement", "x=1;")]
        )

        assert strategy.apply(root, mock_rule, mock_context, "    ") == []
