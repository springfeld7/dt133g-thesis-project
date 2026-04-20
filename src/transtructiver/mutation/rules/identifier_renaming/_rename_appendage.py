"""Private appendage heuristics for RenameIdentifiersRule.

This module is intentionally internal to the rename-identifiers rule.
It derives type-aware suffix tokens from declaration context and formats
renamed identifier text according to language-specific naming conventions.
"""

from ....node import Node
from ..utils.formatter import format_identifier


# Canonical suffix map that collapses equivalent type names to shared tokens.
_TYPE_SUFFIXES = {
    "list": "list",
    "tuple": "tuple",
    "map": "map",
    "set": "set",
    "string": "str",
    **dict.fromkeys(["integer", "float", "double", "number"], "num"),
    "boolean": "flag",
}

# Semantic-only fallback when no concrete type hint can be inferred.
_SEMANTIC_FALLBACK_SUFFIXES = {
    "function_name": "func",
    "class_name": "cls",
    "property_name": "attr",
    "variable_name": "var",
    "parameter_name": "param",
    "argument_name": "arg",
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

    old_text = node.text
    suffix = _infer_suffix(node)
    if old_text.endswith(f"_{suffix}"):
        return old_text

    return format_identifier(node, f"{old_text}_{suffix}", language)


def _infer_suffix(node: Node) -> str:
    """Infer a semantic or type suffix token.

    Delegates first to type-node scanning and falls back to the label-based
    heuristic when no type annotation is present.

    Args:
        node: Identifier node context used for inference.

    Returns:
        Canonical suffix token (for example ``list`` or ``func``).
    """
    return (
        _infer_from_context_type(node)
        or _infer_from_local_declaration_type(node)
        or _infer_from_semantic_label(node)
        or "tmp"
    )


def _infer_from_context_type(node: Node) -> str | None:
    """Return a suffix based solely on the node's context type.

    Args:
        node: Identifier node with a populated ``context_type``.

    Returns:
        Canonical suffix token if the type maps to one, otherwise None.
    """
    if node.context_type:
        for key in _TYPE_SUFFIXES:
            if key in node.context_type:
                return _TYPE_SUFFIXES[key]
    return None


def _infer_from_semantic_label(node: Node) -> str | None:
    """Return a suffix based solely on the node's semantic label.

    Args:
        node: Identifier node with a populated ``semantic_label``.

    Returns:
        Canonical suffix token if the label maps to one, otherwise None.
    """
    if node.semantic_label and node.semantic_label in _SEMANTIC_FALLBACK_SUFFIXES:
        return _SEMANTIC_FALLBACK_SUFFIXES[node.semantic_label]
    return None


def _infer_from_local_declaration_type(node: Node) -> str | None:
    """Infer a type token from nearby declaration structure when context_type is missing."""
    parent = node.parent
    if not parent:
        return None

    # typed_parameter(identifier, type(...))
    if parent.type == "typed_parameter" and parent.children and parent.children[0] is node:
        for sibling in parent.children[1:]:
            inferred = _extract_type_from_node(sibling)
            if inferred:
                return _TYPE_SUFFIXES.get(inferred, inferred)

    # assignment(left, '=', right)
    if parent.type == "assignment" and parent.children and parent.children[0] is node:
        for sibling in parent.children[1:]:
            if sibling.type == "operator" and sibling.text == "=":
                continue
            inferred = _extract_type_from_node(sibling)
            if inferred:
                return _TYPE_SUFFIXES.get(inferred, inferred)

    return None


def _extract_type_from_node(node: Node) -> str | None:
    """Extract canonical type token from a node or its subtree."""
    for child in node.traverse():
        inferred = _to_canonical_type(child.type) or _to_canonical_type(child.text)
        if inferred:
            return inferred

    return None


def _to_canonical_type(raw: str | None) -> str | None:
    """Map raw type/literal text to canonical type keys used in _TYPE_SUFFIXES."""
    if not raw:
        return None

    token = raw.strip().lower().strip("\"'")
    if not token:
        return None

    aliases = {
        "str": "string",
        "int": "integer",
        "bool": "boolean",
        "dict": "map",
        "array": "list",
    }

    token = aliases.get(token, token)

    if token in _TYPE_SUFFIXES:
        return token

    return None
