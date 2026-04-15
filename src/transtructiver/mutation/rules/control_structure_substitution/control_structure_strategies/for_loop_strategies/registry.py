"""Registry for language-specific 'for' loop substitution strategies.

This module provides a centralized mechanism for retrieving the correct
strategy implementation based on the language of the CST.

Each supported language must have a corresponding strategy registered here.
"""

from typing import Dict, Type

from .base_for_loop_strategy import BaseForLoopStrategy
from .cpp_strategy import CppForLoopStrategy
from .java_strategy import JavaForLoopStrategy
from .python_strategy import PythonForLoopStrategy


# Mapping of language identifiers to their corresponding strategy classes
_STRATEGY_REGISTRY: Dict[str, Type[BaseForLoopStrategy]] = {
    "java": JavaForLoopStrategy,
    "cpp": CppForLoopStrategy,
    "python": PythonForLoopStrategy,
}


def get_for_loop_strategy(language: str) -> BaseForLoopStrategy:
    """
    Retrieves the appropriate 'for' loop strategy for a given language.

    Args:
        language (str): The language identifier (e.g., 'java', 'cpp', 'python').

    Returns:
        BaseForLoopStrategy: An instance of the corresponding strategy.

    Raises:
        ValueError: If no strategy exists for the specified language.
    """
    if not language:
        raise ValueError("Language must be provided.")

    normalized = language.lower().strip()

    strategy_cls = _STRATEGY_REGISTRY.get(normalized)
    if not strategy_cls:
        raise ValueError(f"No for-loop strategy registered for language: {language}")

    return strategy_cls()
