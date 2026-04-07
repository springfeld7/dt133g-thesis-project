"""comment_normalization.py

A concrete MutationRule that normalizes comment content by replacing it with
a fixed placeholder while preserving comment delimiters.

This module defines CommentNormalizationRule, which traverses a CST to identify
comment nodes based on semantic labels, and updates their text content to a standardized format.

Each modification generates a MutationRecord capturing the original coordinates
and content of the comment for downstream verification and manifest generation.
"""

from typing import List

from ..identifier_renaming._scope_manager import ScopeManager
from ..mutation_rule import MutationRule, MutationRecord
from ....node import Node
from ._replacement_generator import ReplacementGenerator


class CommentNormalizationRule(MutationRule):
    """
    Mutation rule that normalizes code comments to a fixed placeholder format.

    This rule traverses the CST, identifies comment nodes (line or block comments)
    based on semantic labels, and replaces their content with a standardized
    template, while preserving the original comment delimiters.

    Each normalization produces a MutationRecord capturing the original and new
    content, coordinates, and context for downstream verification and manifesting.
    """

    # Comment markers of supported programming languages
    _DELIMITERS = {
        "line_comment": ["//", "#", "--"],
        "block_comment": [("/**", "*/"), ("/*", "*/"), ('"""', '"""'), ("'''", "'''")],
    }

    # Semantic labels that introduce a fresh naming scope during traversal.
    _SCOPE_LABELS = {
        "root",
        "function_scope",
        "class_scope",
        "loop_scope",
        "condition_scope",
        "block_scope",
    }

    def __init__(self, level: int = 0):
        """Initialize the rule configuration.

        Args:
            level: Replace strategy level.
        """
        super().__init__()
        self.level = level
        # Precompute replacement strategy used during apply().
        self._replacer = ReplacementGenerator(level)
        self._scope = ScopeManager()

    # CLI rule name (used by the auto-discovery in cli.py).
    rule_name = "comment-normalization"

    def _is_scope_node(self, node: Node) -> bool:
        """Return whether the node introduces a new identifier scope."""
        return node.semantic_label in self._SCOPE_LABELS

    def _is_comment_node(self, node: Node) -> bool:
        """Return whether the node is a comment node."""
        return bool(node.semantic_label) and "comment" in node.semantic_label

    def _resolve_ancestor(self, node: Node) -> Node:
        """Return the ancestor to use for content replacement"""
        parent = node.parent
        current = self._scope.current()

        if not parent or not current:
            raise ValueError("Comment node has no scope or parent. Check errors during parsing.")

        if not any(label == parent.semantic_label for label in self._SCOPE_LABELS):
            return self._resolve_ancestor(parent)

        # If parent is a block and node it's first child -> set ancestor from outer scope
        # to correctly identify Java/C++ comments on the same line as the block starts
        # Example:
        # void aFunc() {   // this comment is first child of a block and share start row
        # }
        if all("block_scope" in s for s in [current, parent.semantic_label]):
            if parent.start_point[0] == node.start_point[0]:
                return self._resolve_ancestor(parent)

        return parent

    def _resolve_content(self, node: Node) -> str:
        """Return the new content for a comment."""
        ancestor = self._resolve_ancestor(node)

        return self._replacer.get_replacement(node, ancestor)

    def apply(self, root: Node) -> List[MutationRecord]:
        """Apply the CommentNormalization mutation rule to the CST.

        This method recursively traverses the tree rooted at `root`,
        identifies comment nodes based on semantic labels,
        and updates their text to a normalized format while preserving delimiters.

        Each modification generates a MutationRecord with the original coordinates
        and content of the comment.

        Args:
            root (Node): The root node of the CST to mutate.

        Returns:
            List[MutationRecord]: A list of all modifications performed,
                each containing the original coordinates and content of the normalized comment.
        """
        records: List[MutationRecord] = []

        if root is None:
            return records

        language = root.language.lower() if root.language else None
        if language is None:
            raise ValueError(
                "No language found on root node. "
                "Set root.language before applying CommentNormalizationRule."
            )

        self._scope.reset()
        stack: list[tuple[Node, bool]] = [(root, False)]

        while stack:
            node, is_exit = stack.pop()

            if is_exit:
                # Exit markers let a single iterative walk mirror recursive scope unwinding.
                self._scope.exit_scope()
                continue

            if self._is_scope_node(node):
                self._scope.enter_scope()
                node_type = node.semantic_label if node.semantic_label else node.type
                self._scope.declare(node_type, "")
                stack.append((node, True))

            if not self._is_comment_node(node):
                for child in reversed(node.children):
                    stack.append((child, False))
                continue

            label = node.semantic_label
            delimiters = self._DELIMITERS.get(label) if label else None

            if delimiters:
                # Handle nested block comments (needed for Python triple-quoted docstrings)
                if len(node.children) > 0:
                    for sub in node.children:
                        if sub.type == "string_content":
                            new_text = self._resolve_content(node)
                            if new_text == node.text:
                                continue
                            records.append(self._update_text(sub, new_text))
                # Handle leaf node comments (line comments or block comments without nested content)
                else:
                    new_text = self._format(node, delimiters)
                    if new_text == node.text:
                        continue
                    records.append(self._update_text(node, new_text))

            for child in reversed(node.children):
                stack.append((child, False))

        self._scope.reset()
        return records

    def _format(self, node: Node, delimiters: list) -> str:
        """Replaces text while preserving comment delimiters.

        This method checks the provided text against known comment delimiters and
        returns a new string that maintains the original delimiters but replaces the content
        with a fixed placeholder. It handles both line and block comment styles.

        Args:
            node (Node): The original comment node to be normalized.
            delimiters (list): A list of delimiters to check against,
                which can include line commentprefixes or block comment start/end pairs.
        Returns:
            str: The normalized comment text with original delimiters and replaced content.
        """
        if not node.text:
            return ""

        new_text = self._resolve_content(node)

        stripped_text = node.text.strip()
        for d in delimiters:
            if isinstance(d, tuple):  # Block Style
                start, end = d

                if stripped_text.startswith(start) and stripped_text.endswith(end):
                    lines = new_text.splitlines()
                    new_lines = []
                    prefix = " "

                    if start == "/**":
                        prefix = " *"

                    for line in lines:
                        new_lines.append(f"{prefix}{line.strip()}")

                    return f'{start}\n{"\n".join(new_lines)}\n{end}'

            else:  # Line Style
                if stripped_text.startswith(d):
                    return f"{d} {new_text}"

        return node.text

    def _update_text(self, node: Node, new_text: str) -> MutationRecord:
        """Updates the text of a node and returns a MutationRecord describing the change.

        This method modifies the node's text to the new value and creates a MutationRecord
        that captures the original coordinates and content of the node before the update.

        Args:
            node (Node): The node to update.
            new_text (str): The new text to set on the node.
        Returns:
            MutationRecord: A record of the mutation performed,
            including the node's original coordinates and content.
        """
        return self.record_reformat(node, new_text)
