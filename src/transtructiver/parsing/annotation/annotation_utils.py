"""Shared traversal and filtering utilities for language annotators.

This module provides small helpers used by language-specific annotators to
inspect and filter CST nodes. Functions are intentionally lightweight and
describe common predicates used by the annotation flow (scope detection,
identifier candidacy, and type-like heuristics).

The goal is to centralize small, well-documented helpers so language
annotators remain concise and share common behavior.
"""

from typing import Mapping

from ...node import Node


def meaningful_children(node: Node) -> list[Node]:
    """Return child nodes excluding whitespace and newline tokens."""
    return [child for child in node.children if child.type not in {"whitespace", "newline"}]


def named_children(node: Node) -> list[Node]:
    """Return child nodes that correspond to grammar rules.

    The tree may contain unnamed / lexical children (punctuation, string
    fragments). Use ``is_named`` to obtain only grammar-produced children
    which are the ones relevant for semantic labeling.

    Args:
        node: Node whose named children are requested.

    Returns:
        List of named child nodes.
    """
    return [child for child in node.children if child.is_named]


def is_scope_node(node: Node, type_map: Mapping[str, str] | None = None) -> bool:
    """Determine whether ``node`` acts as a scoping construct.

    Scopes are constructs that introduce a new binding environment (for
    example, functions, classes, and namespaces). The helper uses two
    heuristics:
      - A node with no parent is considered the root scope.
      - If label ends with ``_scope`` the node is treated as a scope.
        Additionally, a node whose ``field`` is ``body`` is considered
        a scope child.

    Args:
        node: Node under inspection.
        type_map: Mapping from node.type to semantic labels (usually the
            annotator's ``direct_type_labels`` or ``direct_field_labels``).

    Returns:
        True if the node is considered to introduce or represent a scope.
    """
    if not node.parent:
        return True

    label = ""

    if node.semantic_label:
        label = node.semantic_label
    elif type_map:
        label = type_map.get(node.type)

    return node.field == "body" or (label.endswith("_scope") if label else False)


def is_type_like_node(node: Node) -> bool:
    """Heuristic check: does ``node`` represent, or contain, a type?"""
    if node.field and "type" in node.field:
        return True

    if "type" in node.type:
        return True

    return False


def is_identifier_candidate(node: Node) -> bool:
    """Return True when ``node`` is a candidate identifier for labeling."""
    return "identifier" in node.type and not node.field == "type" and bool(node.text)


def is_eligible_for_inference(node: Node) -> bool:
    """Return True for identifier nodes eligible for type/context inference."""
    if not node.text:
        return False
    return not bool(node.builtin)
