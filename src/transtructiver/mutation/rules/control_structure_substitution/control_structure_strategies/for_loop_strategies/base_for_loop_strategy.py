"""
Base abstraction for 'for' loop substitution strategies.

Extends the generic control structure strategy with semantics
specific to 'for' loop transformations.
"""

from abc import abstractmethod

from ..base_control_structure_strategy import BaseControlStructureStrategy
from ....mutation_rule import MutationRecord, MutationRule
from ......node import Node


class BaseForLoopStrategy(BaseControlStructureStrategy):
    """
    Specialization of BaseControlStructureStrategy for 'for' loops.

    This class does not add new methods but serves as a semantic layer
    to group all 'for' loop strategies together.
    """

    def _delete_nodes(self, nodes: list[Node], rule) -> list[MutationRecord]:
        """
        Deletes a list of nodes from their parents and returns the mutation records.

        Args:
            nodes: Nodes to delete.
            rule: MutationRule used to generate records.

        Returns:
            list[MutationRecord]: MutationRecord objects for all deletions.
        """
        records = []

        for n in nodes:
            records.append(rule.record_delete(n.parent, n))

        return records

    @abstractmethod
    def _extract_for_loop_components(self, node: Node) -> tuple:
        """
        Extracts the structural components of a 'for' loop node from a CST.

        This method should be implemented by concrete strategies to identify
        the specific components of a 'for' loop (e.g., initializer, condition, update, body)
        based on the language's syntax.

        Args:
            node: The 'for' loop node.

        Returns:
            A tuple containing the extracted components (e.g., for_node, initializer, condition, update, body).
        """
        pass

    @abstractmethod
    def _clean_for_loop_header(self, node: Node, rule) -> list[MutationRecord]:
        """
        Deletes excess formatting nodes from the 'for' loop header, leaving only the essential components.

        Args:
            node: The 'for' loop node.
            rule: MutationRule used to generate records.
        Returns:
            list[MutationRecord]: MutationRecord objects for all deletions.
        """
        pass
