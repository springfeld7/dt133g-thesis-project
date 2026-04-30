"""
Tests for IndentationUtils.

Validates indentation detection logic from CST whitespace nodes.
"""

from typing import cast
import unittest

from src.transtructiver.mutation.rules.utils.indentation_util import IndentationUtils
from .....src.transtructiver.node import Node


class FakeNode:
    def __init__(self, type_, start_point=(0, 0), text="", children=None):
        self.type = type_
        self.start_point = start_point
        self.text = text
        self._children = children or []

    def traverse(self):
        yield self
        for child in self._children:
            yield from child.traverse()


def _ws(text, col=0):
    """Create whitespace node."""
    return FakeNode("whitespace", start_point=(0, col), text=text)


def _id_node():
    """Create non-whitespace node."""
    return FakeNode("identifier", start_point=(0, 0), text="x")


class IndentationUtilsTest(unittest.TestCase):

    def setUp(self):
        """Create utils instance."""
        self.utils = IndentationUtils()

    def _run(self, *nodes):
        """Build root node and run detection."""
        root = FakeNode("root", children=list(nodes))
        return self.utils.detect_indent_unit(cast(Node, root))

    def test_detects_spaces(self):
        """Detect space indentation."""
        self.assertEqual(self._run(_ws("    ", 0)), "    ")

    def test_detects_tabs(self):
        """Detect tab indentation."""
        self.assertEqual(self._run(_ws("\t\t", 0)), "\t\t")

    def test_ignores_non_zero_column(self):
        """Ignore non-zero column whitespace."""
        self.assertEqual(self._run(_ws("    ", 2)), "")

    def test_ignores_invalid_whitespace(self):
        """Ignore non whitespace-only text."""
        self.assertEqual(self._run(_ws("  a ", 0)), "")

    def test_prefers_first_match(self):
        """Return first valid indentation."""
        self.assertEqual(
            self._run(_ws("    ", 0), _ws("\t\t", 0)),
            "    ",
        )

    def test_no_whitespace_nodes(self):
        """Return empty when none found."""
        self.assertEqual(self._run(_id_node()), "")
