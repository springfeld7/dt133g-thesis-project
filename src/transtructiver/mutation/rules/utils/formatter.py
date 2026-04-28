"""Identifier formatting helpers used by renaming mutation rules.

This module encapsulates language-specific identifier naming conventions
and provides utilities to convert a proposed replacement token into a
properly styled identifier for the target language (for example, Python's
snake_case vs Java/C++ camelCase/PascalCase for type names).
"""

from operator import indexOf
from typing import Callable
from ....node import Node


# Per-language title detection rules. When an identifier is considered a
# title (for example a class name) it should use PascalCase formatting.
_IS_TITLE = {
    "python": lambda n: n.semantic_label == "class_name",
    "java": lambda n: n.semantic_label == "class_name",
    "cpp": lambda n: n.semantic_label == "class_name",
}


def _is_title(node: Node, language: str) -> bool:
    """Return True when the node should be formatted as a title.

    Uses the small ``_IS_TITLE`` map to avoid hardcoding language-specific
    heuristics throughout the codebase.
    """
    return _IS_TITLE.get(language, lambda n: False)(node)


def _format_snake_case(words: list[str]) -> str:
    """Format an identifier using snake_case style."""
    if all(len(w) == 1 for w in words):
        return "".join(words).lower()
    return "_".join(words).lower()


def _format_camel_case(words: list[str]) -> str:
    """Format an identifier using camelCase style.

    The first word is left lower-case; subsequent words are capitalized.
    """
    new_name = []
    for w in words:
        if indexOf(words, w) == 0:
            new_name.append(w.lower())
        else:
            new_name.append(w.capitalize())
    return "".join(new_name)


def _format_pascal_case(words: list[str]) -> str:
    """Format an identifier using PascalCase style (TitleCase)."""
    return "".join([w.capitalize() for w in words])


# Explicit per-language name formatters. New languages can be added here
# without modifying the formatting entrypoint.
_LANGUAGE_FORMATTERS: dict[str, Callable[[list[str]], str]] = {
    "python": _format_snake_case,
    **dict.fromkeys(["java", "cpp"], _format_camel_case),
}


def format_identifier(node: Node, new_text: str, language: str) -> str:
    """Format a new identifier according to ``language`` and ``node``.

    The function splits the provided ``new_text`` on underscores and
    applies a language-appropriate formatter. If the identifier is a
    title-like symbol (e.g. a class name) the returned string uses
    PascalCase irrespective of language-specific formatter choices.

    Args:
        node: Node being renamed; used to detect title semantics.
        new_text: New identifier text (commonly a single token or
            underscore-separated words).
        language: Language key resolved from the root node (e.g. "python",
            "java", "cpp").

    Returns:
        Formatted identifier string that follows the conventions for the
        selected language and semantic kind.
    """
    words = split_words(new_text)

    if not words:
        return ""

    if words[0] == "destruct":
        if _is_title(node, language):
            return words[-1].upper()
        return words[-1].lower()

    if _is_title(node, language):
        return _format_pascal_case(words)

    return _LANGUAGE_FORMATTERS.get(language, _format_camel_case)(words)


def split_words(text: str) -> list[str]:
    """Split identifier text into words based on camelCase and underscores.

    Args:
        text: The identifier text to split.
        language: The programming language (used for language-specific splitting rules).

    Returns:
        List of extracted words.
    """
    words = []
    current_word = ""

    for i, char in enumerate(text):
        if char == "_":
            if current_word:
                words.append(current_word)
                current_word = ""
        elif i > 0 and char.isupper() and text[i - 1].islower():
            # CamelCase boundary: uppercase after lowercase
            if current_word:
                words.append(current_word)
            current_word = char
        else:
            current_word += char

    if current_word:
        words.append(current_word)

    return words
