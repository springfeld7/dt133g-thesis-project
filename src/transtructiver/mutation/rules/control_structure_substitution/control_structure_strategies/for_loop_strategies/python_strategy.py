"""Python 'for' loop substitution strategy.

Transforms:

    for x in iterable:
        body

into:

    _iter = iter(iterable)
    while True:
        try:
            x = next(_iter)
        except StopIteration:
            break
        body
"""

from typing import List

from ......node import Node
from .....mutation_context import MutationContext
from ....mutation_rule import MutationRecord, MutationRule
from .base_for_loop_strategy import BaseForLoopStrategy


class PythonForLoopStrategy(BaseForLoopStrategy):
    """
    Strategy for transforming Python 'for' loops into 'while' loops.
    """

    def is_valid(self, node: Node) -> bool:
        """
        Validates Python 'for' loops.

        Excludes:
            - for-else constructs

        Args:
            node (Node): CST node.

        Returns:
            bool: True if valid.
        """
        if node.type != "for_statement":
            return False

        # Exclude for-else
        for child in node.children:
            if child.type == "else_clause":
                return False

        return True

    def apply(
        self,
        node: Node,
        context: MutationContext,
    ) -> List[MutationRecord]:
        """
        Transforms Python 'for' loop into a 'while' loop using iterator protocol.

        Args:
            node (Node): The 'for' node.
            context (MutationContext): Context.
            rule (MutationRule): Rule instance.

        Returns:
            List[MutationRecord]: Transformation record.
        """
        text = node.text

        # naive extraction (can be improved later)
        header, body = text.split(":", 1)
        header = header.replace("for", "", 1).strip()

        var, iterable = [x.strip() for x in header.split(" in ")]

        new_code = f"""
_iter = iter({iterable})
while True:
    try:
        {var} = next(_iter)
    except StopIteration:
        break
{body}
""".strip()

        return [
            rule.record_replace(
                node.start_point,
                node.end_point,
                new_code,
                "while_loop",
            )
        ]
