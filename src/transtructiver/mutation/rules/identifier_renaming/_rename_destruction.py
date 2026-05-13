"""Destructured-name generator for identifier renaming.

This module constructs a compact, type-aware replacement name for
identifiers when the "destruction" naming strategy is selected. The
constructed token is intentionally terse (single-letter canonical codes)
to disrupt naming conventions.

The strategy observes the node's ``context_type`` where available and
maps common type keywords to short codes. When no type information can
be inferred the generator falls back to the generic ``x`` code.
"""

from typing import Optional

from ...mutation_context import MutationContext
from ....node import Node
from ..utils.formatter import format_identifier


# Compact type code map used by the destruction strategy.
_TYPE_MAP = {
    "list": "l",
    "arr": "a",
    "dict": "d",
    "str": "s",
    "int": "i",
    "num": "n",
    "bool": "b",
    "func": "f",
    "df": "d",
    "conn": "c",
}


def _build_destructed_name(node: Node, language: str, _: Optional[MutationContext]) -> str:
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

    original = node.text
    new_text = "destruct"

    for hint, char in _TYPE_MAP.items():
        if hint in original:
            return format_identifier(node, f"{new_text}_{char}", language)

    return format_identifier(node, f"{new_text}_{original[0]}", language)
