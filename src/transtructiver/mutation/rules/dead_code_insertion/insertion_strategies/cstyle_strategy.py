"""C-Style Insertion Strategy

Handles code insertion for brace-delimited languages (C, C++, Java, etc.).

This strategy:
* **Calculates Indentation**: Determines the required leading whitespace by
    analyzing the block's vertical structure.
* **Validates Insertion Points**: Checks gaps between nodes to ensure
    insertions only occur at valid logical breaks (e.g., after newlines).
* **Defines Terminal Conditions**: Identifies statements that end execution
    within a block (like returns or breaks) to manage insertion flow.
"""

from typing import Optional

from .insertion_strategy import InsertionStrategy
from .....node import Node


class CStyleInsertionStrategy(InsertionStrategy):
    """
    Structural strategy for C-Style (C++/Java) code insertion.

    Guards against inserting code outside of braces or on the same line
    as a brace/semicolon.
    """

    def get_indent_prefix(self, node: Node) -> str:
        """
        Iterates through children to find a newline node to guarantee there is
        vertical structure, then looks for the first 'whitespace' node
        and extracts the text as the indentation prefix.

        Args:
            node (Node): The 'block_scope' node being analyzed.

        Returns:
            str: The whitespace string to be used as a prefix for inserted code,
                 or empty string if no suitable prefix can be determined.
        """
        for child in node.children:
            if child.text and child.type == "whitespace":
                return child.text

        return ""

    def is_valid_container(self, node: Node) -> bool:
        """
        Validates that the node is a block scope suitable for insertion.

        Checks that the block scope is not a single-line block.

        Args:
            node (Node): The node to validate.

        Returns:
            bool: True if the node is a valid container for insertion.
        """
        has_newline = any(child.type == "newline" for child in node.children)
        if not has_newline:
            return False

        return True

    def is_valid_gap(self, current: Node, preceding: Optional[Node]) -> bool:
        """
        Validates if the current node is a valid insertion point in brace-delimited languages.

        Args:
            current (Node): The node after the gap.
            preceding (Node | None): The node before the gap.

        Returns:
            bool: True if following a newline and not before a closing brace.
        """
        # Never insert before the opening or closing brace of a block
        if current.type == "{" or current.type == "}":
            return False

        # Must follow a newline, but don't jam code right before the '}'
        if preceding and preceding.type == "whitespace":
            return True

        return False

    def is_terminal(self, node: Node) -> bool:
        """
        Identifies terminal statements for C-style blocks.

        Args:
            node (Node): The node to evaluate.

        Returns:
            bool: True for return, break, continue, or throw.
        """
        terminal_types = {
            "return_statement",
            "break_statement",
            "continue_statement",
        }
        return node.type in terminal_types
