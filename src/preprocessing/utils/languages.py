"""Utility for retrieving tree-sitter Language instances.

This module provides a centralized way to get the tree-sitter Language
for different programming languages by string key.
"""

import tree_sitter_python
import tree_sitter_java
import tree_sitter_cpp

from tree_sitter import Language
from typing import Dict, Any


# Registry of supported languages and their corresponding modules/functions.
# This can be expanded as more languages are added.
_LANGUAGE_MAP: Dict[str, Any] = {
    "python": tree_sitter_python.language(),
    "java": tree_sitter_java.language(),
    "cpp": tree_sitter_cpp.language(),
}


def get_language(language_name: str) -> Language:
    """Retrieve the tree-sitter Language for the given string key.

    Args:
        language_name (str): The name of the language (e.g., "python").

    Returns:
        tree_sitter.Language: The Language instance for the specified language.

    Raises:
        ValueError: If the language is not supported or not installed.
    """
    lang_key = language_name.lower()
    language = None
    if lang_key in _LANGUAGE_MAP:
        language: object = _LANGUAGE_MAP[lang_key]

    if language:
        return Language(language)

    raise ValueError(f"Language '{language_name}' is not supported or missing required package.")
