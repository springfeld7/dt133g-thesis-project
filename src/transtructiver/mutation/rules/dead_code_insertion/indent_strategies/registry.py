"""Indentation strategies Registry

This module maps language identifiers to their respective 
indentation discovery strategies.
"""

from .python_strategy import PythonIndent
from .cstyle_strategy import CStyleIndent


# Global Registry of Language Strategies
_STRATEGY_MAP = {
    "python": PythonIndent(),
    "java": CStyleIndent(),
    "cpp": CStyleIndent(),
}


def get_indentation_prefix(node, language: str) -> str | None:
    """
    Retrieves the indentation prefix for any supported language.

    Args:
        node (Node): The 'block_scope' node being analyzed.
        language (str): The language identifier (e.g., 'python', 'java').

    Returns:
        str | None: The whitespace string used as a prefix, or None
                    if a reliable indentation cannot be determined.
    """
    # Normalize language name and fetch strategy
    # Defaulting to CStyleIndent as the most common pattern
    strategy = _STRATEGY_MAP.get(language.lower())

    if not strategy:
        return None

    return strategy.get_prefix(node)
