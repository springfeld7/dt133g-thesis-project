"""Semantic annotation dispatcher and unified label definitions.

This module provides the central registry for semantic labels used across
all language annotators (Python, Java, C++). It dispatches nodes to the
appropriate language-specific annotator based on the root node type, and
defines unified semantic labels for consistent identifier classification
across different programming languages.

Key concepts:
- Unified semantic labels enable consistent treatment of equivalent constructs
  across languages (e.g., Python's function_definition, Java's method_declaration,
  and C++'s function_definition all map to 'function_scope')
- Language-specific naming ancestor labels map declaration contexts to semantic
  categories (e.g., 'parameter_name', 'class_name', 'function_name')
- Lazy-loaded annotators prevent circular imports
"""

import os

from collections.abc import Callable

from ...node import Node
from .annotators import BaseAnnotator


# Registry mapping language keys to their annotate functions.
_ANNOTATOR_REGISTRY: dict[str, Callable[[Node, dict[str, str]], Node]] = {}


def register_annotator(language: str, func: Callable[[Node, dict[str, str]], Node]) -> None:
    """Register a language annotator function.

    Args:
        language (str): Language key (e.g., 'python', 'java', 'cpp').
        func (Callable[[Node], Node]): Annotator function for the language.
    """
    _ANNOTATOR_REGISTRY[language.lower()] = func


def annotate(root: Node) -> Node:
    """Annotate a node tree with semantic labels based on language.

    Main entry point for semantic annotation. Determines the programming
    language from ``root.language`` and dispatches to the registered
    language-specific annotator. Each identifier node receives a
    semantic_label describing its declaration context (e.g., variable_name,
    function_name, parameter_name).

    Args:
        root (Node): The root node of the syntax tree to annotate.
            ``root.language`` must be set (e.g., "python", "java", "cpp").
        profile (object): The language profile (e.g., LanguageProfile) for builtin/type checking.

    Returns:
        Node: The same root node with semantic_label attributes set on all
            identifier and scope-defining nodes throughout the tree.

    Raises:
        ValueError: If ``root.language`` is not present or unsupported.
    """

    explicit_language = getattr(root, "language", None)
    if not explicit_language:
        raise ValueError(
            "No language found on root node. Set root.language before calling annotate()."
        )

    language = BaseAnnotator.normalize_language_key(explicit_language)

    # Load the correct LanguageProfile for the language
    from .builtin_checker import make_profile_from_files

    base_dir = os.path.join(os.path.dirname(__file__), f"profiles/{language}")
    profile = make_profile_from_files(language, base_dir)

    annotator = BaseAnnotator.for_language(language)
    return annotator.annotate(root, profile)
