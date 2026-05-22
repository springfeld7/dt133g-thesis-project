"""Format-only replacement strategy for comment normalization."""

import re
import unicodedata
import textwrap

from ....node import Node

_LINE_DELIMITERS = ("//", "#", "--")
_BLOCK_DELIMITERS = (("/**", "*/"), ("/*", "*/"), ('"""', '"""'), ("'''", "'''"))


def _is_normalized_character(character: str) -> bool:
    """Allow letters, numbers, whitespace, and ordinary punctuation."""
    if character in {"\ufe0f", "\u200d"}:
        return False

    category = unicodedata.category(character)
    return category.startswith(("L", "N", "P")) or character.isspace()


def _normalize_written_content(text: str, preserve_leading: bool = False) -> str:
    """Remove non-text symbols and normalize spacing while preserving newlines.

    If `preserve_leading` is True, leading whitespace on each non-empty line
    is preserved (useful for block comments where indentation matters).
    """

    def _normalize_line(line: str) -> str:
        without_symbols = "".join(
            character for character in line if _is_normalized_character(character)
        )
        normalized = re.sub(r"\s+", " ", without_symbols).strip()
        punct_normalized = re.sub(r"\s+([!?.,;:)\]\}])", r"\1", normalized)

        if preserve_leading:
            leading = re.match(r"^(\s*)", line)
            leading_text = leading.group(1) if leading else ""
            if not punct_normalized:
                return f"{leading_text}"
            return f"{leading_text}{punct_normalized}"

        if not punct_normalized:
            return ""

        return punct_normalized

    if preserve_leading:
        lines = text.split("\n")
    else:
        lines = textwrap.wrap(text, width=80)
    if len(lines) <= 1:
        return _normalize_line(text)

    normalized_lines = [_normalize_line(line) for line in lines]
    return "\n".join(line for line in normalized_lines if line)


def _replace_format_only(node: Node, _ancestor: Node) -> str:
    """Return the comment's written content with normalized spacing and symbols."""
    if not node.text:
        if not node.semantic_label == "block_comment":
            return ""

    if node.semantic_label == "block_comment" and len(node.children) > 0:
        new_text = "".join(n.text for n in node.children if n.text)
    else:
        new_text = node.text if node.text else ""

    label = node.semantic_label or ""

    if label.startswith("line_"):
        for delimiter in _LINE_DELIMITERS:
            if new_text.startswith(delimiter):
                return _normalize_written_content(new_text[len(delimiter) :].lstrip())

    if label.startswith("block_"):
        for start, end in _BLOCK_DELIMITERS:
            if new_text.startswith(start) and new_text.endswith(end):
                content = new_text[len(start) : -len(end)]
                return f"\n{_normalize_written_content(content, preserve_leading=True)}"

    return new_text
