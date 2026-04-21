"""Utility functions for renaming identifiers by length compression.

Provides functions to abbreviate and compress identifiers to shorter forms.
"""

from ....node import Node
from ..utils.formatter import split_words, format_identifier


def _build_abbreviated_name(node: Node, language: str) -> str:
    """Build a shortened identifier name based on length.

    Args:
        node: The node containing the identifier text and semantic label.
        language: The programming language (python, java, cpp, etc.).

    Returns:
        The formatted shortened identifier, or empty string if node has no text.
    """

    if not node.text:
        return ""

    words = split_words(node.text)
    new_text = _compress_words(words, language)

    return format_identifier(node, new_text, language)


def _compress_words(words: list[str], language: str) -> str:
    """Compress a list of words into a shorter identifier."""
    if not words:
        return ""

    chars = []

    if len(words) == 1:
        word = words[0].lower()
        # Abbreviate single word to 2-3 chars based on length
        if len(word) <= 3:
            chars.append(word)
        elif len(word) <= 6:
            chars.append(word[0])
            chars.append(word[-1])
        else:
            chars.append(word[0])
            chars.append(word[len(word) // 2])
            chars.append(word[-1])

    elif len(words) <= 3:
        # For 2-3 words, use first letter of each
        for w in words:
            chars.append(w[0].lower())
    else:
        # For more than 3 words, use first letter of first 3 words
        for w in words[:3]:
            chars.append(w[0].lower())

    return "".join(chars) if language == "python" else "_".join(chars)
