"""Unit tests for JavaForLoopStrategy in control structure substitution mutation rules.

Focuses on validating Java-specific node extraction (local_variable_declaration)
and the transformation logic for converting for-loops to while-loops, including scoping considerations.
"""

import pytest
from unittest.mock import MagicMock, patch

from transtructiver.node import Node
from transtructiver.mutation.mutation_context import MutationContext
from transtructiver.mutation.rules.mutation_rule import MutationRecord, MutationRule
from transtructiver.mutation.rules.control_structure_substitution.control_structure_strategies.for_loop_strategies.java_strategy import (
    JavaForLoopStrategy,
)


class TestJavaForLoopStrategy:
    """
    Comprehensive test suite for JavaForLoopStrategy.
    Covers base node manipulation, C-style utilities, and Java-specific scoping.
    """

    @pytest.fixture
    def strategy(self):
        return JavaForLoopStrategy()

    @pytest.fixture
    def mock_rule(self):
        rule = MagicMock(spec=MutationRule)
        # Mocking return values to act like records
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
        ctx._id_gen = 1000

        def next_id():
            ctx._id_gen += 1
            return ctx._id_gen

        ctx.next_id.side_effect = next_id
        return ctx

    # -------------------------------------------------------------------------
    # BASE LOGIC TESTS (BaseControlStructureStrategy)
    # -------------------------------------------------------------------------

    def test_base_node_insertion(self, strategy, mock_context, mock_rule):
        """Verifies low-level node creation and parent linking."""
        parent = Node((0, 0), (0, 0), "parent")
        point = (5, 5)

        record = strategy._insert_node(
            mock_context, parent, point, "test_type", "test_text", 0, mock_rule
        )

        assert len(parent.children) == 1
        assert parent.children[0].type == "test_type"
        assert parent.children[0].text == "test_text"
        assert parent.children[0].parent == parent
        assert record["new_type"] == "test_type"

    def test_get_indent_logic(self, strategy):
        """Ensures indentation is correctly pulled from preceding whitespace."""
        ws = Node((1, 0), (1, 4), "whitespace", "    ")
        target = Node((1, 4), (1, 8), "for_statement")
        parent = Node((0, 0), (2, 0), "block", children=[ws, target])
        target.parent = parent

        assert strategy._get_indent(target) == "    "
        assert strategy._get_indent(ws) == ""  # First child or no WS sibling should be empty

    # -------------------------------------------------------------------------
    # C-STYLE UTILITY TESTS (CstyleForLoopStrategy)
    # -------------------------------------------------------------------------

    def test_cstyle_header_cleaning(self, strategy, mock_rule):
        """Tests removal of semicolons and whitespace from within ( )."""
        semi = Node((0, 10), (0, 11), ";", ";")
        ws = Node((0, 11), (0, 12), "whitespace", " ")
        loop = Node(
            (0, 0),
            (0, 20),
            "for_statement",
            children=[
                Node((0, 0), (0, 3), "for"),
                Node((0, 4), (0, 5), "("),
                semi,
                ws,
                Node((0, 15), (0, 16), ")"),
            ],
        )
        semi.parent = ws.parent = loop

        records = strategy._clean_for_loop_header(loop, mock_rule)
        assert len(records) == 2
        mock_rule.record_delete.assert_any_call(loop, semi)

    def test_effective_body_detection(self, strategy):
        """Ensures we don't transform loops that are effectively empty."""
        empty_block = Node(
            (0, 0), (0, 2), "block", children=[Node((0, 0), (0, 1), "{"), Node((0, 1), (0, 2), "}")]
        )
        full_block = Node(
            (0, 0),
            (0, 5),
            "block",
            children=[
                Node((0, 0), (0, 1), "{"),
                Node((0, 1), (0, 4), "stmt"),
                Node((0, 4), (0, 5), "}"),
            ],
        )

        assert strategy._has_effective_body(empty_block) is False
        assert strategy._has_effective_body(full_block) is True

    # -------------------------------------------------------------------------
    # JAVA SPECIFIC TESTS (JavaForLoopStrategy)
    # -------------------------------------------------------------------------

    def test_normalize_variable_declarations(self, strategy):
        """Tests splitting 'int i=0, j=1' into two statements."""
        type_n = Node((0, 0), (0, 3), "primitive_type", "int")
        type_n.semantic_label = "type_name"
        v1 = Node((0, 4), (0, 7), "variable_declarator", "i=0")
        v2 = Node((0, 9), (0, 12), "variable_declarator", "j=1")
        decl = Node((0, 0), (0, 12), "local_variable_declaration", children=[type_n, v1, v2])

        sources = strategy._normalize_init_nodes([decl])
        assert sources == ["int i=0;", "int j=1;"]

    def test_apply_transformation_with_scoping(self, strategy, mock_rule, mock_context):
        """
        Integration test: for(int i=0; i<1; i++) { body; }
        Should result in extra braces { ... } to wrap the init and while loop.
        """
        # Build complex tree
        for_kw = Node((0, 0), (0, 3), "for", "for")
        decl = Node((0, 5), (0, 15), "local_variable_declaration", "int i=0")
        cond = Node((0, 17), (0, 20), "binary_expression", "i<1")
        upd = Node((0, 22), (0, 25), "update_expression", "i++")

        b_open, b_stmt, b_close = (
            Node((1, 0), (1, 1), "{"),
            Node((1, 2), (1, 7), "body;"),
            Node((1, 8), (1, 9), "}"),
        )
        body = Node((1, 0), (1, 9), "block", children=[b_open, b_stmt, b_close])

        root = Node(
            (0, 0),
            (1, 9),
            "for_statement",
            children=[
                for_kw,
                Node((0, 3), (0, 4), "("),
                decl,
                Node((0, 15), (0, 16), ";"),
                cond,
                Node((0, 20), (0, 21), ";"),
                upd,
                Node((0, 25), (0, 26), ")"),
                body,
            ],
        )
        root.parent = Node((0, 0), (0, 0), "module", children=[root])

        with patch.object(strategy, "_apply_indent_reformat", return_value=[]) as mock_reformat:
            records = strategy.apply(root, mock_rule, mock_context, "    ", level=0)

            # 1. Check that "{" was inserted for scope block
            assert any(rec.get("new_type") == "{" for rec in records)

            # 2. Check for-while substitution
            # The second argument to record_substitute is the OLD type ("for")
            mock_rule.record_substitute.assert_called_with(for_kw, "for")

            # 3. Verify the node itself was actually updated
            assert for_kw.type == "while"
            assert for_kw.text == "while"

            mock_reformat.assert_called_once()

    def test_infinite_loop_empty_condition(self, strategy, mock_rule, mock_context):
        """Verifies that an empty condition in for(;;) results in 'true'."""
        close_paren = Node((0, 10), (0, 11), ")", ")")
        root = Node(
            (0, 0),
            (0, 20),
            "for_statement",
            children=[
                Node((0, 0), (0, 3), "for"),
                Node((0, 4), (0, 5), "("),
                Node((0, 5), (0, 6), ";"),
                Node((0, 7), (0, 8), ";"),
                close_paren,
                Node(
                    (0, 12),
                    (0, 20),
                    "block",
                    children=[
                        Node((0, 12), (0, 13), "{"),
                        Node((0, 14), (0, 15), "s"),
                        Node((0, 19), (0, 20), "}"),
                    ],
                ),
            ],
        )

        strategy._insert_default_true_condition(root, mock_context, mock_rule)

        mock_rule.record_insert.assert_called_once_with(
            point=(1001, -1),
            insertion_point=close_paren.start_point,
            new_text="true",
            new_type="binary_expression",
        )

    def test_bailout_on_missing_braces(self, strategy, mock_rule, mock_context):
        """Java allows for(...) stmt; - we should skip if no block braces exist."""
        root = Node(
            (0, 0),
            (0, 10),
            "for_statement",
            children=[
                Node((0, 0), (0, 3), "for"),
                Node((0, 5), (0, 10), "expression_statement", "x=1;"),  # No block
            ],
        )

        assert strategy.apply(root, mock_rule, mock_context, "    ", level=0) == []

    def test_indent_reformat_only_affects_column_zero(self, strategy, mock_rule):
        """Ensures we don't break mid-line alignment."""
        ws_bol = Node((1, 0), (1, 2), "whitespace", "  ")
        ws_mid = Node((1, 10), (1, 11), "whitespace", " ")

        mock_node = MagicMock()
        mock_node.traverse.return_value = [ws_bol, ws_mid]

        strategy._apply_indent_reformat(mock_node, "    ", mock_rule)

        # Should only record reformat for the node at column 0
        mock_rule.record_reformat.assert_called_once_with(ws_bol, "      ")
