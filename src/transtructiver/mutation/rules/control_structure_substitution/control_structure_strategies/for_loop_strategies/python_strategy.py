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
        self, node: Node, rule: MutationRule, context: MutationContext, indent_unit: str, level: int
    ) -> List[MutationRecord]:
        """
        Transforms Python 'for' loop into a 'while' loop using iterator protocol.

        Args:
            node (Node): The traditional for-loop node.
            rule (MutationRule): Rule instance used for recording mutations.
            context (MutationContext): Shared mutation context.
            indent_unit (str): The indentation unit for the language.
            level (int): The transformation level to apply.

        Returns:
            List[MutationRecord]: Transformation record.
        """
        records = []
        for_node, item, in_node, iterable, body = self._extract_for_loop_components(node)

        # Ensure all components are present and that body is not empty
        if not all([for_node, item, in_node, iterable, body]) or len(body.children) == 0:
            return []

        # Build source code strings for item and iterable
        item_src = item.to_code().strip()
        iterable_src = iterable.to_code().strip()

        # Delete and substitute header components
        records.extend(self._delete_nodes([item, iterable], rule))
        records.extend(self._clean_for_loop_header(node, rule))
        records.append(self._substitute(for_node, "while", "while", rule))
        records.append(self._substitute(in_node, "true", "True", rule))

        # Find indentation for the new nodes to be inserted
        indent = ""
        target_node = node.parent.parent
        if target_node:
            ws_node = None
            for child in reversed(target_node.children):
                if child.type == "whitespace":
                    ws_node = child
                    break
            if not ws_node:
                return []  # Can't find indentation, abort transformation

            # Get what will be needed for iterator initialization
            indent = ws_node.text

        for_node_idx = node.children.index(for_node)
        iter_var = self._get_unique_iter_name(context, level)

        # Insert iterator initialization before the transformed while loop
        records.extend(
            self._insert_segments(
                context,
                node,
                for_node.start_point,
                [
                    ("whitespace", indent),
                    ("newline", "\n"),
                    ("initializer", f"{iter_var} = iter({iterable_src})"),
                ],
                for_node_idx,
                rule,
            )
        )

        insert_idx = self._find_body_insertion_index(node)
        if insert_idx == -1:
            return []  # Can't find insertion point, abort transformation

        # Insert the try-except and next() call at the start of the loop body
        records.extend(
            self._insert_segments(
                context,
                node,
                node.children[insert_idx].start_point,
                [
                    ("whitespace", indent + indent_unit),
                    ("newline", "\n"),
                    ("synthetic_break", "break"),
                    ("whitespace", indent + indent_unit + indent_unit),
                    ("newline", "\n"),
                    ("synthetic_except", "except StopIteration:"),
                    ("whitespace", indent + indent_unit),
                    ("newline", "\n"),
                    ("synthetic_assignment", f"{item_src} = next({iter_var})"),
                    ("whitespace", indent + indent_unit + indent_unit),
                    ("newline", "\n"),
                    ("synthetic_try", "try:"),
                ],
                insert_idx,
                rule,
            )
        )
        return records

    def _extract_for_loop_components(self, node: Node):
        """
        Extracts the structural components of a Python for-loop node from a CST.

        This method traverses the direct children of the provided node and identifies
        the key components of a for-loop structure:
        for node, item node, iterable node and body node.

        Args:
            node (Node): The CST node expected to represent a for-loop structure.

        Returns:
            tuple:
                - for_node (Optional[Node]): The node representing the entire for-loop.
                - item (Optional[Node]): The node representing the loop variable(s).
                - in_node (Optional[Node]): The node representing the 'in' keyword (used as an anchor).
                - iterable (Optional[Node]): The node representing the iterable being looped over.
                - body (Optional[Node]): The node representing the body of the loop.
        """
        ignored = ["whitespace", "newline", ":", "comment"]
        content = [c for c in node.children if c.type not in ignored]

        # content[0] = for
        # content[1] = item (x)
        # content[2] = in  <-- This is our new anchor!
        # content[3] = iterable (list)
        # content[4] = body (block)
        for_node, item, in_node, iterable, body, *_ = content + [None] * 5

        return for_node, item, in_node, iterable, body

    def _clean_for_loop_header(self, node: Node, rule) -> list[MutationRecord]:
        """
        Removes syntactic noise inside a for-loop header and records deletions.

        It generates and returns mutation records for the deletion of these nodes.

        Args:
            node (Node): The for-loop CST node whose header is being cleaned.
            rule: MutationRule used to generate deletion records.

        Returns:
            list[MutationRecord]: A list of mutation records for the deletions.
        """
        header_formatting = []

        for child in node.children:
            # Once we hit the colon, the header is officially over.
            if child.type == ":":
                break

            if child.type in ("whitespace", "newline"):
                header_formatting.append(child)

        # Preserve the first formatting node (the space after 'while')
        # Record deletions for everything else in the header area
        to_delete = header_formatting[1:]

        return [rule.record_delete(node, n) for n in to_delete]

    def _get_unique_iter_name(self, context: MutationContext, level: int) -> str:
        """
        Generates a unique iterator variable name to avoid collisions with existing identifiers.

        This method checks the 'iter_var' base name against the set of identifiers already
        present in the source code. If a collision is detected, it appends an incrementing
        numeric suffix until a unique name is found. The resulting name is added to the
        context's taken names to prevent collisions with subsequent transformations in the
        same execution.

        Args:
            context (MutationContext): The current mutation context containing the set
                of all identifiers ('taken_names') found during the initial CST traversal.
            level (int): The transformation level, which can be used to influence naming conventions.

        Returns:
            str: A unique string identifier (e.g., 'iter_var', 'iter_var_1', 'iter_var_2')
                guaranteed not to exist in the current scope.
        """
        base = "iter_var"

        # Check if the base name is available
        if base not in context.taken_names:
            context.taken_names.add(base)  # Reserve it for this loop
            return base

        # Iterate with a counter until a unique name is found
        counter = 1
        while f"{base}_{counter}" in context.taken_names:
            counter += 1

        unique_name = f"{base}_{counter}"
        context.taken_names.add(unique_name)  # Reserve it for the next loop in the file

        return unique_name

    def _get_unique_iter_name(self, context: MutationContext, level: int) -> str:
        """
        Generate a unique iterator variable name based on transformation level.

        Naming strategy:
            - Level 0–1: "iter", "iter_1", ...
            - Level 2: "it", "it1", "it2", ...
            - Level 3: single letters ("a"–"z"), then "i1", "i2", ...

        Args:
            context (MutationContext): Holds all identifiers already used in the code.
            level (int): Controls how compact the variable name should be.

        Returns:
            str: A unique iterator variable name.
        """
        taken = context.taken_names

        def reserve(name: str) -> str:
            """Reserve and return a name."""
            taken.add(name)
            return name

        # ----------------------------
        # LEVEL 0–1: "iter", "iter1", ...
        # ----------------------------
        if level <= 1:
            base = "iter"
            if base not in taken:
                return reserve(base)

            counter = 1
            while (candidate := f"{base}{counter}") in taken:
                counter += 1
            return reserve(candidate)

        # ----------------------------
        # LEVEL 2: "it", "it1", ...
        # ----------------------------
        if level == 2:
            if "it" not in taken:
                return reserve("it")

            counter = 1
            while (candidate := f"it{counter}") in taken:
                counter += 1
            return reserve(candidate)

        # ----------------------------
        # LEVEL >=3: single-letter, then fallback
        # ----------------------------
        for c in "abcdefghijklmnopqrstuvwxyz":
            if c not in taken:
                return reserve(c)

        counter = 1
        while (candidate := f"i{counter}") in taken:
            counter += 1
        return reserve(candidate)

    def _find_body_insertion_index(self, node: Node) -> int:
        """
        Locates the index of the first non-formatting node following the colon.

        This method identifies the structural boundary (the colon) and returns
        the index of the first subsequent child that represents actual code
        content, bypassing leading whitespace or newlines within the body.

        Args:
            node (Node): The 'for' loop node.

        Returns:
            int: The index within the node's children for the insertion.
        """
        passed_colon = False

        for i, child in enumerate(node.children):
            if child.type == ":":
                passed_colon = True
                continue

            if passed_colon and child.type not in ("whitespace", "newline"):
                return i

        # Fallback for an empty body where everything after the colon is formatting
        return -1
