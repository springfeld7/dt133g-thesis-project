"""
Indentation detection utilities for CST processing.

This module provides logic for detecting indentation patterns from a
concrete syntax tree by scanning leading whitespace nodes.
"""

from ....node import Node


class IndentationUtils:
    """
    Utilities for detecting and working with indentation patterns in a CST.
    """

    @staticmethod
    def detect_indent_unit(root: Node) -> str:
        """
        Scans the tree for the first whitespace node that starts at column 0
        and has a length greater than 0.

        This method is used to auto-detect the indentation unit, fallbacks to 4 spaces if none is found.

        Args:
            root (Node): The root of the CST to scan for indentation patterns.

        Returns:
            str: The detected indentation unit (e.g. "    " for 4 spaces) or a default if none found.
        """
        # Traverse to find the first 'indentation' whitespace with a length > 0
        for node in root.traverse():
            if node.type == "whitespace" and node.start_point[1] == 0:
                if node.text and all(c in (" ", "\t") for c in node.text):
                    return node.text
        return ""  # Fallback to empty string if no indentation pattern is found, which will be treated as no indentation.
