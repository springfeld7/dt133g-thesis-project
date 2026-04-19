"""Shared traversal and filtering utilities for language annotators."""

from typing import Mapping

from ...node import Node


def meaningful_children(node: Node) -> list[Node]:
    """Return child nodes excluding whitespace and newline tokens."""
    return [child for child in node.children if child.type not in {"whitespace", "newline"}]


def named_children(node: Node) -> list[Node]:
    """Return child nodes that correspond to grammar rules, excluding spaces and string literals."""
    return [child for child in node.children if child.is_named]


def is_scope_node(node: Node, type_map: Mapping[str, str]) -> bool:
    if not node.parent:
        return True

    label = type_map.get(node.type)
    return node.field == "body" or (label.endswith("_scope") if label else False)


def is_type_like_node(node: Node) -> bool:
    if node.field and "type" in node.field:
        return True

    if "type" in node.type:
        return True

    # if not node.field and node.parent:
    #     if node.parent.type == "type":
    #         return True

    return False


def is_identifier_candidate(node: Node) -> bool:
    return "identifier" in node.type and not node.field == "type" and bool(node.text)


def is_eligible_for_inference(node: Node) -> bool:
    if not node.text:
        return False
    return not bool(node.builtin)
