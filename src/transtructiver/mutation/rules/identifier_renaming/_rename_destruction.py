"""Destructured-name generator for identifier renaming.

This module constructs a compact, type-aware replacement name for
identifiers when the "destruction" naming strategy is selected. The
constructed token is intentionally terse (single-letter canonical codes)
to disrupt naming conventions.

The strategy observes the node's ``context_type`` where available and
maps common type keywords to short codes. When no type information can
be inferred the generator falls back to the generic ``x`` code.
"""

from ....node import Node
from ..utils.formatter import format_identifier


# Compact type code map used by the destruction strategy.
_TYPE_MAP = {
    "set": "c",
    "tuple": "t",
    "list": "l",
    "map": "m",
    "string": "s",
    "number": "n",
    "boolean": "b",
}


def _build_destructed_name(node: Node, language: str) -> str:
    """Build a short, type-hinted replacement name for *node*.

    Args:
        node: Identifier node being renamed.
        language: Target language key used for final formatting.

    Returns:
        A formatted identifier string (for example
        ``c`` for variables or ``C`` for class names), or the
        empty string when *node.text* is empty.
    """
    if not node.text:
        return ""

    context_type = node.context_type if node.context_type else ""
    code = _TYPE_MAP.get(context_type, "x")

    new_text = f"destruct_{code}"
    return format_identifier(node, new_text, language)
