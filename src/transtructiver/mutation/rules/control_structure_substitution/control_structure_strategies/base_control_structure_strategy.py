"""
Base abstraction for language-specific control structure substitution strategies.

Each concrete strategy must define how to:
1. Identify valid control structure nodes
2. Transform them into equivalent structures
"""

from abc import ABC, abstractmethod
from typing import List

from .....node import Node
from ....mutation_context import MutationContext
from ...mutation_rule import MutationRecord, MutationRule


class BaseControlStructureStrategy(ABC):
    """
    Abstract base class for control structure substitution strategies.

    Concrete implementations must provide logic for identifying valid
    control structures and rewriting them into equivalent forms.

    Methods:
        is_valid(node): Determines if the node is a valid control structure candidate.
        apply(node, context, rule): Performs the transformation.
    """

    @abstractmethod
    def is_valid(self, node: Node) -> bool:
        """
        Determines whether a node is a valid candidate for transformation.

        Args:
            node (Node): The CST node.

        Returns:
            bool: True if the node can be transformed, False otherwise.
        """
        pass

    @abstractmethod
    def apply(self, node: Node, context: MutationContext, indent_unit: str) -> List[MutationRecord]:
        """
        Transforms a valid control structure into an equivalent form.

        Args:
            node (Node): The CST node representing the control structure.
            context (MutationContext): Context for mutation tracking.
            indent_unit (str): The indentation unit for the language.

        Returns:
            List[MutationRecord]: Records describing the transformation.
        """
        pass

    def _get_indent(self, node: Node) -> str:
        """
        Retrieves the indentation whitespace for a given node by examining its preceding siblings.

        Assumes indenation is represented by a whitespace node immediately preceding the target node.

        Args:
            node (Node): The node for which to find indentation.

        Returns:
            str: The indentation whitespace, or an empty string if none found.
        """
        parent = node.parent
        if not parent:
            return ""

        idx = parent.children.index(node)
        if idx == 0:
            return ""

        prev = parent.children[idx - 1]
        if prev.type != "whitespace":
            return ""

        return prev.text or ""
