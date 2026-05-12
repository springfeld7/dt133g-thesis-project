"""
Core node structure for representing Abstract and Concrete Syntax Trees.

This module defines the fundamental Node class used throughout the project
to represent hierarchical code structures.
"""

import json
from typing import Any, Iterator, List, Optional


class Node:
    """
    Represents a node in an Abstract Syntax Tree (AST) or Concrete Syntax Tree (CST).

    A Node is a fundamental building block for representing program structure as a tree.
    Each node has a type (e.g., "identifier", "binary_expression"), optional child nodes,
    and optional text content (for leaf nodes like tokens).
    """

    def __init__(
        self,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
        type: str,
        text: Optional[str] = None,
        children: Optional[List[Node]] = None,
    ) -> None:
        """
        Initialize a new Node.

        Args:
            start_point (tuple[int, int]): Starting position as (row, column).
            end_point (tuple[int, int]): Ending position as (row, column).
            type (str): The node type/category (e.g., "identifier", "function_definition").
            text (str, optional): The raw token text for leaf nodes. None for non-leaf nodes.
            children (list): List of child Node objects. Represents the structure of the tree.
            semantic_label (str, optional): Semantic label for the node (e.g., "function_name", "variable_name").
        """
        self.start_point = start_point
        self.end_point = end_point
        self.type = type
        self.text = text
        self.children = children or []
        # Links populated during adaptation and annotation for tree navigation.
        self.parent: Optional[Node] = None
        self.semantic_label: Optional[str] = None
        self.context_type: Optional[str] = None
        self.field: Optional[str] = None
        self.language: Optional[str] = None
        self.builtin: bool = False
        self.is_named: bool = True

    def add_child(self, child: Node) -> None:
        """
        Add a child node to this node.

        Args:
            child (Node): The child node to add.
        """
        self.children.append(child)

    def remove_child(self, child: Node) -> None:
        """
        Remove a child node from this node.

        Args:
            child (Node): The child node to remove.

        Raises:
            ValueError: If the node is not a child of this node.
        """
        self.children.remove(child)

    def traverse(self) -> Iterator[Node]:
        """
        Yield all nodes in the tree using preorder traversal.

        Preorder traversal visits the current node before visiting its children.
        This is useful for visiting nodes in a top-down order.

        Yields:
            Node: Each node in the tree in preorder sequence.
        """
        yield self
        for child in self.children:
            yield from child.traverse()

    def traverse_up(self) -> Iterator[Node]:
        """
        Yields ancestors of the current node, starting from the parent
        up to the root of the tree.

        Yields:
            Node: Each parent node in the tree.
        """
        curr = self.parent
        while curr is not None:
            yield curr
            curr = curr.parent

    def clone(self, parent: Optional[Node] = None) -> Node:
        """
        Creates a deep copy of this node and its entire subtree.

        Args:
            parent (Node, optional): Parent of the cloned node.

        Returns:
            Node: Root of the cloned subtree.
        """
        new_node = Node(
            start_point=self.start_point,
            end_point=self.end_point,
            type=self.type,
            text=self.text,
            children=[],
        )

        new_node.semantic_label = self.semantic_label
        new_node.context_type = self.context_type
        new_node.field = self.field
        new_node.language = self.language
        new_node.builtin = self.builtin
        new_node.is_named = self.is_named
        new_node.parent = parent

        new_node.children = [child.clone(new_node) for child in self.children]

        return new_node

    def to_dict(self) -> dict[str, Any]:
        """Serialize this node and its subtree to plain Python data.

        Parent links are intentionally excluded because they are derived from
        the child structure and would create cycles in a serialized format.
        """

        return {
            "start_point": list(self.start_point),
            "end_point": list(self.end_point),
            "type": self.type,
            "text": self.text,
            "semantic_label": self.semantic_label,
            "context_type": self.context_type,
            "field": self.field,
            "language": self.language,
            "builtin": self.builtin,
            "is_named": self.is_named,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], parent: Optional[Node] = None) -> Node:
        """Rebuild a node tree from :meth:`to_dict` output."""
        node = cls(
            start_point=tuple(payload["start_point"]),
            end_point=tuple(payload["end_point"]),
            type=payload["type"],
            text=payload.get("text"),
            children=[],
        )
        node.semantic_label = payload.get("semantic_label")
        node.context_type = payload.get("context_type")
        node.field = payload.get("field")
        node.language = payload.get("language")
        node.builtin = bool(payload.get("builtin", False))
        node.is_named = bool(payload.get("is_named", True))
        node.parent = parent

        for child_payload in payload.get("children", []):
            child = cls.from_dict(child_payload, node)
            node.children.append(child)

        return node

    def to_json(self) -> str:
        """Serialize this node tree to a JSON string."""

        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> Node:
        """Deserialize a JSON string produced by :meth:`to_json`."""

        return cls.from_dict(json.loads(payload))

    def __repr__(self) -> str:
        """
        Return a string representation of the node.

        Returns:
            str: A string showing the node type and text (if present).
        """
        return f"Node(type={self.type}, text={self.text})"

    def to_code(self) -> str:
        """
        Reconstruct source code from the CST by collecting leaf-node text in
        pre-order traversal order.

        Leaf nodes are identified as nodes with no children that carry a text
        value. This is used by the CLI to produce the mutated code string for
        the augmented dataset (FR-9).

        Returns:
            str: The reconstructed source code string.
        """
        return "".join(
            node.text for node in self.traverse() if not node.children and node.text is not None
        )

    def pretty(self, indent: int = 0) -> None:
        """
        Print a human-readable tree representation of this node and its children.

        This method recursively prints the tree structure with proper indentation.
        Each level of nesting is indented by 2 spaces.

        Example:
            >>> node = Node("function_definition", text="add")
            >>> node.pretty()
            function_definition: add
        """
        prefix = "  " * indent
        if (self.type != "newline") and (self.type != "whitespace"):
            if self.field:
                line = f"{prefix}('{self.field}') {self.type}"
            else:
                line = f"{prefix}{self.type}"

            if self.text:
                line += f": {self.text}"

            if self.semantic_label:
                line += f"  [{self.semantic_label}]"
            if self.context_type:
                line += f"<{self.context_type}>"

            print(line)

        for child in self.children:
            child.pretty(indent + 1)
