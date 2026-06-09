"""Format-only replacement strategy for comment normalization."""

import re
import unicodedata

from unidecode import unidecode

from ....node import Node

_LINE_DELIMITERS = ("//", "#")
_BLOCK_DELIMITERS = (("/**", "*/"), ("/*", "*/"), ('"""', '"""'), ("'''", "'''"))


def _normalize_written_content(text: str, preserve_leading: bool = False) -> str:
    """Normalize unicode and spacing"""

    def normalize_unicode(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(unidecode(c) for c in s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s

    def _normalize_line(line: str) -> str:
        original = line
        line = normalize_unicode(line)
        line = re.sub(r"[\u200B-\u200D\uFEFF\\]", "", line)
        line = re.sub(r"(\s+:)|(:{3,})", ":", line)
        line = re.sub(r"(\s+;)|(;{2,})", ";", line)
        line = re.sub(r"(\s+!)|(!{2,})", "!", line)
        line = re.sub(r"(\s+\?)|(\?{2,})", "?", line)
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"[-+,.]{2,}", "", line)
        line = re.sub(r"[-=+,.]{3,}", "", line)
        line = re.sub(r"^\s+[-=+,.]+$", "", line)

        if preserve_leading:
            m = re.match(r"^(\s*)", original)
            leading = m.group(1) if m else ""
            stripped = line.strip()
            return leading + stripped
        else:
            stripped = line.strip()
            line = stripped if stripped else ""

        return line

    lines = text.split("\n")
    normalized_lines = [_normalize_line(line) for line in lines]
    new_text = "\n".join(normalized_lines)
    new_text = re.sub(r"^\n+", "\n", new_text)

    return new_text.strip()


def _replace_format_only(node: Node, _ancestor: Node) -> str:
    """Return the comment's written content with normalized spacing and symbols."""
    if not node.text:
        if not node.semantic_label == "block_comment":
            return ""

    if node.semantic_label == "block_comment" and len(node.children) > 0:
        new_text = "".join(n.text for n in node.children if n.text).lstrip()
    else:
        new_text = node.text if node.text else ""

    label = node.semantic_label or ""

    if label.startswith("line_"):
        for delimiter in _LINE_DELIMITERS:
            if new_text.startswith(delimiter):
                return _normalize_written_content(new_text[len(delimiter) :])

    if label.startswith("block_"):
        for start, end in _BLOCK_DELIMITERS:
            if new_text.startswith(start) and new_text.endswith(end):
                content = new_text[len(start) : -len(end)]

                return _normalize_written_content(content, preserve_leading=True)

    return new_text
