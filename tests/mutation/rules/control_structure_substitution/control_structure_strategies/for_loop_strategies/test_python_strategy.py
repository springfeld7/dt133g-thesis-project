"""Unit tests for PythonForLoopStrategy.

Focuses on the Python-specific iterator protocol transformation and 
unique variable name generation for iterators.
"""

import pytest
from unittest.mock import MagicMock, patch

from transtructiver.node import Node
from transtructiver.mutation.mutation_context import MutationContext
from transtructiver.mutation.rules.mutation_rule import MutationRule
from transtructiver.mutation.rules.control_structure_substitution.control_structure_strategies.for_loop_strategies.python_strategy import (
    PythonForLoopStrategy,
)


class TestPythonForLoopStrategy:
    """
    Test suite for PythonForLoopStrategy.
    Covers name collision, for-else exclusion, and the while-true transformation.
    """

    @pytest.fixture
    def strategy(self):
        return PythonForLoopStrategy()

    @pytest.fixture
    def mock_rule(self):
        rule = MagicMock(spec=MutationRule)
        rule.record_insert.side_effect = lambda **kwargs: {"type": "insert", **kwargs}
        rule.record_substitute.side_effect = lambda node, old_type: {
            "type": "substitute",
            "old": old_type,
        }
        rule.record_delete.side_effect = lambda parent, node: {"type": "delete", "node": node}
        return rule

    @pytest.fixture
    def mock_context(self):
        ctx = MagicMock(spec=MutationContext)
        ctx.taken_names = set()
        ctx._id_gen = 3000
        ctx.next_id.side_effect = lambda: setattr(ctx, "_id_gen", ctx._id_gen + 1) or ctx._id_gen
        return ctx

    def _n(self, n_type, text="", children=None):
        node = Node((0, 0), (0, len(text)), n_type, text)
        if children:
            node.children = children
            for child in children:
                child.parent = node
        return node

    # -------------------------------------------------------------------------
    # COMPONENT & VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_is_valid_excludes_for_else(self, strategy):
        """Python for-else cannot be safely transformed into a simple while-true."""
        normal_loop = self._n("for_statement", children=[self._n("for"), self._n("block")])
        else_loop = self._n(
            "for_statement", children=[self._n("for"), self._n("block"), self._n("else_clause")]
        )

        assert strategy.is_valid(normal_loop) is True
        assert strategy.is_valid(else_loop) is False

    def test_extract_python_components(self, strategy):
        """Verifies the 5-way split: for, item, in, iterable, body."""
        for_kw = self._n("for", "for")
        item = self._n("identifier", "x")
        in_kw = self._n("in", "in")
        iterable = self._n("identifier", "items")
        body = self._n("block", "print(x)")

        root = self._n(
            "for_statement",
            children=[
                for_kw,
                self._n("whitespace", " "),
                item,
                self._n("whitespace", " "),
                in_kw,
                self._n("whitespace", " "),
                iterable,
                self._n(":"),
                body,
            ],
        )

        f, i, in_n, it, b = strategy._extract_for_loop_components(root)
        assert f == for_kw
        assert i == item
        assert in_n == in_kw
        assert it == iterable
        assert b == body

    # -------------------------------------------------------------------------
    # UNIQUE NAME GENERATION
    # -------------------------------------------------------------------------

    def test_get_unique_iter_name_collision(self, strategy, mock_context):
        """Ensures 'iter_var' increments if the name is already in 'taken_names'."""
        mock_context.taken_names.add("iter_var")
        mock_context.taken_names.add("iter_var_1")

        name = strategy._get_unique_iter_name(mock_context)

        assert name == "iter_var_2"
        assert "iter_var_2" in mock_context.taken_names

    # -------------------------------------------------------------------------
    # APPLY / TRANSFORMATION TESTS
    # -------------------------------------------------------------------------

    def test_apply_python_transformation(self, strategy, mock_rule, mock_context):
        """
        Integration: for x in items: ...
        Checks for keyword substitutions (for->while, in->True)
        and the insertion of next() and try-except.
        """
        ws_node = self._n("whitespace", "    ")
        for_kw = self._n("for", "for")
        in_kw = self._n("in", "in")
        item = self._n("identifier", "x")
        iterable = self._n("identifier", "items")
        body_stmt = self._n("expression_statement", "pass")
        body = self._n("block", children=[body_stmt])

        loop_node = self._n(
            "for_statement",
            children=[
                for_kw,
                self._n("whitespace", " "),
                item,
                self._n("whitespace", " "),
                in_kw,
                self._n("whitespace", " "),
                iterable,
                self._n(":"),
                body,
            ],
        )

        # Structure for indent lookup
        fake_parent = self._n("expression_statement", children=[loop_node])
        module = self._n("module", children=[ws_node, fake_parent])

        # Use a dummy context with the specific iter_var name
        mock_context.taken_names = set()

        with patch.object(strategy, "_find_body_insertion_index", return_value=8):
            records = strategy.apply(loop_node, mock_rule, mock_context, "    ", level=0)

            # 1. Check Keyword Substitutions
            # We expect the 'old' type to be recorded
            mock_rule.record_substitute.assert_any_call(for_kw, "for")
            mock_rule.record_substitute.assert_any_call(in_kw, "in")

            # 2. Check Iterator Initializer
            # Ensuring it uses the iter_var name
            assert any("iter_var = iter(items)" in str(rec.get("new_text")) for rec in records)

            # 3. Check Body insertions
            # Note: order in records list depends on your _insert_segments implementation
            all_text = "".join(str(rec.get("new_text", "")) for rec in records)
            assert "except StopIteration:" in all_text
            assert "x = next(iter_var)" in all_text
            assert "break" in all_text

    def test_find_body_insertion_point(self, strategy):
        """Ensures we find the first node AFTER the colon for the try-block."""
        colon = self._n(":", ":")
        stmt = self._n("expression_statement", "pass")
        root = self._n(
            "for_statement",
            children=[
                self._n("for"),
                self._n("identifier", "x"),
                colon,
                self._n("newline", "\n"),
                self._n("whitespace", "  "),
                stmt,
            ],
        )

        idx = strategy._find_body_insertion_index(root)
        assert root.children[idx] == stmt
