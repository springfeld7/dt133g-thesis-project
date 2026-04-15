"""C++ 'for' loop substitution strategy.

Transforms C++ for-loops of the form:

    for (init; condition; increment) { body }

into an equivalent while-loop:

    init;
    while (condition) {
        body
        increment;
    }
"""

from typing import List

from ......node import Node
from .....mutation_context import MutationContext
from ....mutation_rule import MutationRecord
from .base_for_loop_strategy import BaseForLoopStrategy


class CppForLoopStrategy(BaseForLoopStrategy):
    """
    Strategy for transforming C++ for-loops into equivalent while-loops.
    """

    def is_valid(self, node: Node) -> bool:
        """
        Checks whether the node represents a C++ for-loop.

        Args:
            node (Node): The CST node.

        Returns:
            bool: True if node is a C++ for-loop.
        """
        return node.type == "for_statement"

    def apply(
        self,
        node: Node,
        context: MutationContext,
    ) -> List[MutationRecord]:
        """
        Transforms a C++ for-loop into a while-loop equivalent.

        Args:
            node (Node): The for-loop node.
            context (MutationContext): Shared mutation context.

        Returns:
            List[MutationRecord]: Replace operation transforming the loop.
        """
        new_code = "hej"

        return [
            self.record_replace(
                node.start_point,
                node.end_point,
                new_code,
                "while_loop",
            )
        ]
