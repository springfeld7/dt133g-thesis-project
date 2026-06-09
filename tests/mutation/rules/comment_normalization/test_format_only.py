"""Unit tests for the format-only comment normalization strategy."""

from transtructiver.node import Node
from transtructiver.mutation.rules.comment_normalization._format_only import (
    _replace_format_only,
)


def _make_node(node_type: str, text: str, semantic_label: str, start=(1, 0), children=[]) -> Node:
    node = Node(start, (1, len(text)), node_type, text=text, children=children)
    node.semantic_label = semantic_label
    return node


class TestReplaceFormatOnly:
    """Direct tests for the level 0 replacement strategy."""

    def test_line_comment_normalizes_spacing_without_rewriting_content(self):
        node = _make_node("line_comment", "//keep this text   ", "line_comment")

        assert _replace_format_only(node, node) == "keep this text"

    def test_block_comment_normalizes_unicode_and_symbols(self):
        content = _make_node(
            "string_content",
            """This function computes ΔE between χ_i and χ_j.  
    Result must be ≥ 0.
    🚀🚀🚀
""",
            "",
        )
        start = _make_node("string_start", '"""', "")
        end = _make_node("string_end", '"""', "")

        node = _make_node(
            "string", "", "block_comment", start=(1, 4), children=[start, content, end]
        )

        assert (
            _replace_format_only(node, node)
            == """This function computes DE between kh_i and kh_j.
    Result must be >= 0."""
        )

    def test_block_comment_normalizes_newlines_and_symbols(self):
        node = _make_node(
            "block_comment",
            "/* hello,\nworld🚀!! */",
            "block_comment",
        )

        assert _replace_format_only(node, node) == "hello,\nworld!"

    def test_line_comment_normalizes_emojis_and_punctuation(self):
        node = _make_node("line_comment", "// Hi there 😀!", "line_comment")

        assert _replace_format_only(node, node) == "Hi there!"

    def test_line_comment_keeps_hyphens_and_underscores(self):
        node = _make_node(
            "line_comment",
            "// keep-this_value, please",
            "line_comment",
        )

        assert _replace_format_only(node, node) == "keep-this_value, please"

    def test_line_comment_preserves_decomposed_letters(self):
        node = _make_node(
            "line_comment",
            "// cafe\u0301 is open",
            "line_comment",
        )

        assert _replace_format_only(node, node) == "cafe is open"

    def test_block_comment_keeps_hyphens_and_underscores(self):
        node = _make_node(
            "block_comment",
            "/* build-step_value, ready */",
            "block_comment",
        )

        assert _replace_format_only(node, node) == "build-step_value, ready"

    def test_empty_comment_text_returns_empty_string(self):
        node = _make_node("line_comment", "", "line_comment")

        assert _replace_format_only(node, node) == ""
