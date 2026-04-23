"""Unit tests for experiments/sample_selection/analysis.py

Validates that SampleAnalyzer correctly computes stylistic and structural
metrics from a manually constructed CST using Node objects.
"""

import pytest

from src.experiments.sample_selection.analysis import SampleAnalyzer
from src.transtructiver.node import Node


# ===== Helpers =====


def make_simple_tree():
    """Return a CST with known counts for loops, identifiers, comments, and whitespace."""
    root = Node((0, 0), (0, 0), "program", children=[])

    # for loop node
    for_node = Node((0, 0), (0, 10), "for_statement")

    # identifiers (2 total)
    id1 = Node((1, 0), (1, 1), "identifier", text="x")
    id2 = Node((2, 0), (2, 1), "identifier", text="y")

    # comment spanning 2 lines
    comment = Node((3, 0), (4, 5), "comment", text="// comment")
    comment.semantic_label = "line_comment"

    # whitespace node (4 spaces + 1 tab = 8 total after weighting)
    whitespace = Node((5, 0), (5, 5), "whitespace", text="    \t")

    root.children = [for_node, id1, id2, comment, whitespace]
    return root


def make_empty_tree():
    """Return an empty CST."""
    return Node((0, 0), (0, 0), "program")


# ===== Fixtures =====


@pytest.fixture
def analyzer():
    """Provide a default SampleAnalyzer instance."""
    return SampleAnalyzer()


@pytest.fixture
def sample_code():
    """Provide a small code snippet with known structure."""
    return "for(i=0;i<1;i++){\n x;\n y;\n}\n"


@pytest.fixture
def sample_tree():
    """Provide a CST with known metric counts."""
    return make_simple_tree()


# ===== Metric Tests =====


def test_char_and_loc_counts(analyzer, sample_code, sample_tree):
    """Verify character count and LOC are calculated correctly."""
    metrics = analyzer.calculate_metrics(sample_code, "java", "HUMAN_GENERATED", sample_tree)

    assert metrics["char_count"] == len(sample_code)
    assert metrics["loc"] == len(sample_code.splitlines())


def test_lloc_counts(analyzer, sample_tree):
    """Verify logical LOC excludes empty lines."""
    code = "a\n\nb\n"
    metrics = analyzer.calculate_metrics(code, "java", "HUMAN_GENERATED", sample_tree)

    assert metrics["lloc"] == 2  # only 'a' and 'b'


def test_for_loop_density(analyzer, sample_code, sample_tree):
    """Verify for loop density is computed correctly."""
    metrics = analyzer.calculate_metrics(sample_code, "java", "HUMAN_GENERATED", sample_tree)

    # 1 for loop / 4 logical lines
    assert metrics["for_loop_density"] == pytest.approx(1 / 4)


def test_identifier_density(analyzer, sample_code, sample_tree):
    """Verify identifier density is computed correctly."""
    metrics = analyzer.calculate_metrics(sample_code, "java", "HUMAN_GENERATED", sample_tree)

    # 2 identifiers / 4 logical lines
    assert metrics["identifier_density"] == pytest.approx(2 / 4)


def test_comment_density(analyzer, sample_code, sample_tree):
    """Verify comment density uses line span correctly."""
    metrics = analyzer.calculate_metrics(sample_code, "java", "HUMAN_GENERATED", sample_tree)

    # comment spans 2 lines, total LOC = 4
    assert metrics["comment_density"] == pytest.approx(2 / 4)


def test_whitespace_ratio(analyzer, sample_code, sample_tree):
    """Verify whitespace ratio accounts for tab expansion."""
    metrics = analyzer.calculate_metrics(sample_code, "java", "HUMAN_GENERATED", sample_tree)

    # whitespace = 8 chars (4 spaces + tab=4 spaces), total length = len(code)
    expected = 8 / len(sample_code)
    assert metrics["whitespace_ratio"] == pytest.approx(expected)


# ===== Edge Cases =====


def test_zero_length_code(analyzer):
    """Verify zero-length code does not cause division errors."""
    tree = make_empty_tree()

    metrics = analyzer.calculate_metrics("", "java", "HUMAN_GENERATED", tree)

    assert metrics["char_count"] == 0
    assert metrics["whitespace_ratio"] == 0.0


def test_empty_tree_metrics(analyzer):
    """Verify metrics on an empty CST return zeros where appropriate."""
    code = ""
    tree = make_empty_tree()

    metrics = analyzer.calculate_metrics(code, "java", "HUMAN_GENERATED", tree)

    assert metrics["for_loop_density"] == 0
    assert metrics["identifier_density"] == 0
    assert metrics["comment_density"] == 0


def test_normalize_lang(analyzer):
    """Verify language normalization handles C++ and casing."""
    assert analyzer._normalize_lang("C++") == "cpp"
    assert analyzer._normalize_lang("Java") == "java"
