from operator import indexOf
from typing import Callable
from ....node import Node


_IS_TITLE = {
    "python": lambda n: n.semantic_label == "class_name",
    "java": lambda n: n.semantic_label == "class_name",
}


def _is_title(node: Node, language: str) -> bool:
    return _IS_TITLE.get(language, lambda n: False)(node)


def _format_snake_case(words: list[str]) -> str:
    """Format an identifier using snake_case style."""
    return "_".join(words)


def _format_camel_case(words: list[str]) -> str:
    """Format an identifier using camelCase style."""
    new_name = []
    for w in words:
        if indexOf(words, w) == 0:
            new_name.append(w)
        else:
            new_name.append(w.capitalize())
    return "".join(new_name)


def _format_pascal_case(words: list[str]) -> str:
    """Format an identifier using PascalCase style."""
    return "".join([w.capitalize() for w in words])


# Explicit per-language name formatters.
# New languages can be added here without modifying _format_new_name.
_LANGUAGE_FORMATTERS: dict[str, Callable[[list[str]], str]] = {
    "python": _format_snake_case,
    **dict.fromkeys(["java", "cpp"], _format_camel_case),
}


def format_identifier(node: Node, new_text: str, language: str) -> str:
    """Format a new identifier from ``new_text``, and ``language``.

    Args:
        new_text: New identifier text.
        language: Language key resolved from root.

    Returns:
        Formatted identifier text according to current language rules.
    """
    words = new_text.split("_")

    if _is_title(node, language):
        return _format_pascal_case(words)

    return _LANGUAGE_FORMATTERS.get(language, _format_camel_case)(words)
