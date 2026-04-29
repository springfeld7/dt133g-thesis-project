"""Format-only replacement strategy for comment normalization."""

import re
import unicodedata

from ....node import Node

_LINE_DELIMITERS = ("//", "#", "--")
_BLOCK_DELIMITERS = (("/**", "*/"), ("/*", "*/"), ('"""', '"""'), ("'''", "'''"))


def _is_normalized_character(character: str) -> bool:
    """Allow letters, numbers, whitespace, and ordinary punctuation."""
    if character in {"\ufe0f", "\u200d"}:
        return False

    category = unicodedata.category(character)
    return category.startswith(("L", "N", "P")) or character.isspace()


def _normalize_written_content(text: str) -> str:
    """Remove non-text symbols and normalize spacing while preserving newlines."""

    def _normalize_line(line: str) -> str:
        without_symbols = "".join(
            character for character in line if _is_normalized_character(character)
        )
        normalized = re.sub(r"\s+", " ", without_symbols).strip()
        return re.sub(r"\s+([!?.,;:)\]\}])", r"\1", normalized)

    lines = text.splitlines()
    if len(lines) <= 1:
        return _normalize_line(text)

    normalized_lines = [_normalize_line(line) for line in lines]
    return "\n".join(line for line in normalized_lines if line)


def _replace_format_only(node: Node, _ancestor: Node) -> str:
    """Return the comment's written content with normalized spacing and symbols."""

    if not node.text:
        return ""

    stripped_text = node.text.strip()
    label = node.semantic_label or ""

    if label.startswith("line_"):
        for delimiter in _LINE_DELIMITERS:
            if stripped_text.startswith(delimiter):
                return _normalize_written_content(stripped_text[len(delimiter) :].lstrip())

    if label.startswith("block_"):
        for start, end in _BLOCK_DELIMITERS:
            if stripped_text.startswith(start) and stripped_text.endswith(end):
                return _normalize_written_content(stripped_text[len(start) : -len(end)].strip())

    return stripped_text
