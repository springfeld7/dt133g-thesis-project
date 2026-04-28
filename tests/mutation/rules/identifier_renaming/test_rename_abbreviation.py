"""Tests for identifier abbreviation renaming utilities."""

from transtructiver.node import Node
from transtructiver.mutation.rules.identifier_renaming._rename_abbreviation import (
    _build_abbreviated_name,
    _compress_words,
)
from transtructiver.mutation.rules.utils.formatter import split_words


def testsplit_words_splits_underscore_and_camelcase():
    """Test splitting identifier into words by underscore and camelCase."""
    assert split_words("foo_bar") == ["foo", "bar"]
    assert split_words("fooBar") == ["foo", "Bar"]
    assert split_words("fooBarBaz") == ["foo", "Bar", "Baz"]
    assert split_words("foo_barBaz") == ["foo", "bar", "Baz"]
    assert split_words("foo") == ["foo"]
    assert split_words("") == []


def test_compress_words_no_words():
    """Test abbreviation returns empty string for empty words list."""
    assert _compress_words([], "python") == ""


def test_compress_words_single_word():
    """Test abbreviation for single-word identifiers."""
    # <=3 chars: keep as is
    assert _compress_words(["id"], "python") == "id"
    # 4-6 chars: first and last
    assert _compress_words(["hello"], "python") == "ho"
    # >6 chars: first, middle, last
    assert _compress_words(["abcdefgh"], "python") == "aeh"


def test_compress_words_two_three_words():
    """Test abbreviation for two or three word identifiers."""
    assert _compress_words(["foo", "bar"], "python") == "fb"
    assert _compress_words(["foo", "bar", "baz"], "python") == "fbb"
    # >3 words
    assert _compress_words(["foo", "bar", "baz", "qux"], "python") == "fbb"


def test_compress_words_language_difference():
    """Test abbreviation output differs by language."""
    # python: joined, others: underscore
    assert _compress_words(["foo", "bar"], "python") == "fb"
    assert _compress_words(["foo", "bar"], "java") == "f_b"


def test_build_abbreviated_name_empty():
    """Test abbreviation returns empty string for empty node text."""
    node = Node((0, 0), (0, 0), "identifier", text="")
    assert _build_abbreviated_name(node, "python") == ""


def test_build_abbreviated_name_long_camelcase():
    """Test abbreviation for long camelCase identifier."""
    node = Node((0, 0), (0, 0), "identifier", text="longVariableName")
    # Should split to ["long", "Variable", "Name"] and compress to "lvn" for python
    result = _build_abbreviated_name(node, "python")
    assert isinstance(result, str)
    assert result == "lvn"


def test_build_abbreviated_name_language():
    """Test abbreviation output for different languages."""
    node = Node((0, 0), (0, 0), "identifier", text="foo_bar_baz")
    py = _build_abbreviated_name(node, "python")
    java = _build_abbreviated_name(node, "java")
    assert py != java
    assert py == "fbb"
    assert java == "fBB"


def test_build_abbreviated_name_short_word():
    """Test abbreviation for very short identifier."""
    node = Node((0, 0), (0, 0), "identifier", text="id")
    assert _build_abbreviated_name(node, "java") == "id"
