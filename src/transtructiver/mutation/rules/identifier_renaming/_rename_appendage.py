"""Private appendage heuristics for RenameIdentifiersRule.

This module is intentionally internal to the rename-identifiers rule.
It derives type-aware suffix tokens from declaration context and formats
renamed identifier text according to language-specific naming conventions.
"""

from ....node import Node
from ....parsing.annotation.annotator import NAMING_ANCESTOR_LABELS


_TYPE_SUFFIXES = {
    **dict.fromkeys(["list"], "list"),
    **dict.fromkeys(["tuple"], "tuple"),
    **dict.fromkeys(["dict", "dictionary"], "dict"),
    **dict.fromkeys(["set"], "set"),
    **dict.fromkeys(["str", "string"], "str"),
    **dict.fromkeys(["int", "integer", "float", "double", "number"], "num"),
    **dict.fromkeys(["bool", "boolean"], "flag"),
}

_NODE_TYPE_SUFFIXES = {
    **dict.fromkeys(["list", "list_comprehension"], "list"),
    **dict.fromkeys(["tuple"], "tuple"),
    **dict.fromkeys(["dictionary", "dictionary_comprehension"], "dict"),
    **dict.fromkeys(["set", "set_comprehension"], "set"),
    **dict.fromkeys(["string", "string_literal"], "str"),
    **dict.fromkeys(["integer", "float", "number"], "num"),
    **dict.fromkeys(["true", "false", "boolean"], "flag"),
}

_SEMANTIC_FALLBACK_SUFFIXES = {
    "function_name": "fn",
    "class_name": "cls",
    "property_name": "attr",
}

_DECLARATION_FIELDS = {
    "left",
    "name",
    "declarator",
    "parameter",
    "pattern",
    "function",
}


def _ancestor_types_for_label(label: str) -> set[str]:
    """Return ancestor node types that map to a semantic naming label.

    Args:
        label: Semantic label to reverse-map (for example ``variable_name``).

    Returns:
        A set of ancestor node types associated with the given label.
    """
    return {
        node_type
        for language_map in NAMING_ANCESTOR_LABELS.values()
        for node_type, semantic_label in language_map.items()
        if semantic_label == label
    }


_ASSIGNMENT_PARENT_TYPES = {
    "assignment",
    "augmented_assignment",
    "assignment_expression",
    "variable_declarator",
    "init_declarator",
}


_LOOP_PARENT_TYPES = {
    "for_statement",
    "enhanced_for_statement",
    "for_range_loop",
}


def _build_appendage_name(node: Node, language: str) -> str:
    """Build renamed identifier text for a node.

    Args:
        node: Identifier node being renamed.
        language: Language key resolved from CST root.

    Returns:
        The formatted identifier text after suffix inference.
    """
    if not node.text:
        return ""

    suffix = _infer_suffix(node, node.text)
    if node.text.endswith(f"_{suffix}"):
        return node.text

    return _format_new_name(node.text, suffix, language)


def _infer_suffix(node: Node, text: str) -> str:
    """Infer a semantic or type suffix token.

    Args:
        node: Identifier node context used for inference.
        text: Current identifier text.

    Returns:
        Canonical suffix token (for example ``list`` or ``fn``), or an empty
        string when no suffix can be inferred.
    """
    inferred_type = None
    parent = node.parent

    if parent and node.semantic_label:
        if parent.type in _ancestor_types_for_label(node.semantic_label):
            for child in _meaningful_children(parent):
                if child is node:
                    continue

                if child.field == "type":
                    inferred_type = _suffix_from_type_node(child)
                    if inferred_type is not None:
                        break

            if inferred_type is None and parent.parent is not None:
                for sibling in parent.parent.children:
                    if sibling is parent:
                        continue

                    if sibling.field == "type":
                        inferred_type = _suffix_from_type_node(sibling)
                        if inferred_type is not None:
                            break

    if inferred_type is not None:
        return inferred_type

    semantic_suffix = _SEMANTIC_FALLBACK_SUFFIXES.get(node.semantic_label or "")
    if semantic_suffix is not None:
        return semantic_suffix

    return ""


def _suffix_from_type_node(type_node: Node) -> str | None:
    """Derive a canonical suffix token from a type annotation node.

    Args:
        type_node: Node representing a type annotation or type-like context.

    Returns:
        Canonical suffix token if resolvable, otherwise None.
    """
    type_text = type_node.text
    if type_text is None:
        for nested in type_node.children:
            if nested.text:
                type_text = nested.text
                break

    if not type_text:
        return None

    lowered = type_text.lower()

    parent = type_node.parent
    if parent is not None and parent.type == "method_reference":
        terminal = lowered.split("::", 1)[0].split(".")[-1].split("<", 1)[0].strip()
    elif parent is not None and parent.type == "qualified_identifier":
        terminal = lowered.rsplit("::", 1)[-1].split(".")[-1].strip()
    else:
        terminal = lowered.split(".")[-1].strip()

    exact_match = _TYPE_SUFFIXES.get(terminal)
    if exact_match is not None:
        return exact_match

    # Fallback: if any known type key appears in the full type text, use it.
    for key in sorted(_TYPE_SUFFIXES.keys(), key=len, reverse=True):
        if key in lowered:
            return _TYPE_SUFFIXES[key]

    return None


def _meaningful_children(node: Node) -> list[Node]:
    """Return children excluding formatting-only nodes.

    Args:
        node: Node whose children should be filtered.

    Returns:
        Child nodes excluding whitespace and newline tokens.
    """
    return [child for child in node.children if child.type not in {"whitespace", "newline"}]


def _format_new_name(text: str, suffix: str, language: str) -> str:
    """Format a new identifier from base ``text``, ``suffix``, and ``language``.

    Args:
        text: Original identifier text.
        suffix: Canonical suffix token.
        language: Language key resolved from root.

    Returns:
        Formatted identifier text according to current language rules.

    Notes:
        Python formatting uses snake_case suffixes, while non-Python
        formatting follows the current fallback behavior implemented here.
    """
    if len(suffix) < 1 or len(language) < 1:
        return text + text[-1]
    elif language.lower() == "python":
        return f"{text}_{suffix}"
    else:
        return text + suffix.capitalize()
