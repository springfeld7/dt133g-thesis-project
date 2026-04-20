"""Utility functions for renaming identifiers by length compression.

Provides functions to abbreviate and compress identifiers to shorter forms.
"""

from transtructiver.node import Node
from transtructiver.mutation.rules.utils.formatter import format_identifier


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

    words = _get_words(node.text)
    new_text = _compress_words(words, language)

    return format_identifier(node, new_text, language)


def _get_words(text: str) -> list[str]:
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


def _compress_words(words: list[str], language: str) -> str:
    """Compress a list of words into a shorter identifier."""
    if not words:
        return ""

    chars = []

    if len(words) == 1:
        word = words[0]
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
            chars.append(w[0])
    else:
        # For more than 3 words, use first letter of first 3 words
        for w in words[:3]:
            chars.append(w[0])

    return "".join(chars) if language == "python" else "_".join(chars)
