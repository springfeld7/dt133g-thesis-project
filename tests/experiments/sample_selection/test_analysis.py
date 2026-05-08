"""Unit tests for experiments/sample_selection/analysis.py

Validates that SampleAnalyzer correctly computes stylistic and structural
metrics from a manually constructed Tree-sitter-shaped node tree.
"""

import pytest
from tree_sitter import Point

from src.experiments.sample_selection.analysis import SampleAnalyzer


# ===== Helpers =====


class MockTSNode:
    """Small Tree-sitter-shaped node used to unit test traversal logic."""

    def __init__(
        self,
        node_type,
        start_point,
        end_point,
        text=None,
        children=None,
        is_named=True,
        start_byte=None,
        end_byte=None,
    ):
        self.type = node_type
        self.start_point = start_point
        self.end_point = end_point
        self.text = text
        self.children = children or []
        self.parent = None
        self.is_named = is_named
        self.start_byte = start_byte if start_byte else None
        self.end_byte = end_byte if end_byte else None

        for child in self.children:
            child.parent = self


def make_node(
    node_type,
    start_point,
    end_point,
    text=None,
    children=None,
    is_named=True,
    start_byte=None,
    end_byte=None,
):
    """Create a mock Tree-sitter node with parent links and byte offsets."""
    return MockTSNode(
        node_type,
        start_point,
        end_point,
        text=text,
        children=children,
        is_named=is_named,
        start_byte=start_byte,
        end_byte=end_byte,
    )


def make_simple_tree():
    """Return a mock CST with known counts for loops, identifiers, and gaps containing whitespace.

    Uses sample code: "for    \t{\nx;\ny;\n}" (with explicit spaces and tab, no indentation)
    Tree has for_statement and 2 identifiers. Whitespace in gap between for and {.
    """
    code = "for    \t{\nx;\ny;\n}"

    # for statement node: bytes 0-3
    for_node = make_node(
        "for_statement",
        Point(0, 0),
        Point(0, 3),
        text="for",
        is_named=True,
        start_byte=0,
        end_byte=3,
    )

    # identifier x: at (1, 0) - after "for    \t{\n"
    # Line 0: "for    \t{\n" = 3 + 4 + 1 + 1 + 1 = 10 bytes
    # Line 1: "x;\n" starts at byte 10, x is at column 0
    id1 = make_node(
        "identifier",
        Point(1, 0),
        Point(1, 1),
        text="x",
        is_named=True,
        start_byte=10,
        end_byte=11,
    )

    # identifier y: at (2, 0)
    # Line 1: "x;\n" = 3 bytes (10-12)
    # Line 2: "y;\n" starts at byte 13, y is at column 0
    id2 = make_node(
        "identifier",
        Point(2, 0),
        Point(2, 1),
        text="y",
        is_named=True,
        start_byte=13,
        end_byte=14,
    )

    # Root program node spanning entire code
    root_byte_end = len(code.encode("utf8"))
    return make_node(
        "program",
        Point(0, 0),
        Point(3, 1),
        children=[for_node, id1, id2],
        is_named=True,
        start_byte=0,
        end_byte=root_byte_end,
    )


def make_empty_tree():
    """Return an empty mock CST."""
    return make_node("program", (0, 0), (0, 0), is_named=True, start_byte=0, end_byte=0)


# ===== Fixtures =====


@pytest.fixture
def analyzer():
    """Provide a default SampleAnalyzer instance."""
    return SampleAnalyzer()


@pytest.fixture
def sample_code():
    """Provide a small code snippet with known structure."""
    return "for    \t{\nx;\ny;\n}"


@pytest.fixture
def sample_tree():
    """Provide a CST with known metric counts."""
    return make_simple_tree()


# ===== Metric Tests =====


def test_char_and_loc_counts(analyzer, sample_code, sample_tree):
    """Verify character count and LOC are calculated correctly."""
    metrics = analyzer.calculate_metrics(sample_code, sample_tree)

    assert metrics["char_count"] == len(sample_code)
    # Code without final newline: "for    \t{\nx;\ny;\n}" has lines when split
    assert metrics["loc"] == len(sample_code.splitlines())


def test_lloc_counts(analyzer, sample_tree):
    """Verify logical LOC excludes empty lines."""
    code = "a\n\nb\n"
    metrics = analyzer.calculate_metrics(code, sample_tree)

    assert metrics["lloc"] == 2  # only 'a' and 'b'


def test_for_loop_density(analyzer, sample_code, sample_tree):
    """Verify for loop density is computed correctly."""
    metrics = analyzer.calculate_metrics(sample_code, sample_tree)

    # 1 for_statement / 4 logical lines
    assert metrics["for_loop_density"] == pytest.approx(1 / 4)


def test_identifier_density(analyzer, sample_code, sample_tree):
    """Verify identifier density is computed correctly."""
    metrics = analyzer.calculate_metrics(sample_code, sample_tree)

    # 2 identifiers / 4 logical lines
    assert metrics["identifier_density"] == pytest.approx(2 / 4)


def test_comment_density(analyzer, sample_code, sample_tree):
    """Verify comment density uses line span correctly."""
    metrics = analyzer.calculate_metrics(sample_code, sample_tree)

    # No comments in sample code
    assert metrics["comment_density"] == 0


def test_whitespace_ratio(analyzer, sample_code, sample_tree):
    """Verify whitespace ratio accounts for tab expansion."""
    metrics = analyzer.calculate_metrics(sample_code, sample_tree)

    # Gap between 'for' and first named node has 4 spaces + 1 tab (8 chars)
    expected = 8 / len(sample_code)
    assert metrics["whitespace_ratio"] == pytest.approx(expected)


# ===== Edge Cases =====


def test_zero_length_code(analyzer):
    """Verify zero-length code does not cause division errors."""
    tree = make_empty_tree()

    metrics = analyzer.calculate_metrics("", tree)

    assert metrics["char_count"] == 0
    assert metrics["whitespace_ratio"] == 0.0


def test_empty_tree_metrics(analyzer):
    """Verify metrics on an empty CST return zeros where appropriate."""
    code = ""
    tree = make_empty_tree()

    metrics = analyzer.calculate_metrics(code, tree)

    assert metrics["for_loop_density"] == 0
    assert metrics["identifier_density"] == 0
    assert metrics["comment_density"] == 0
