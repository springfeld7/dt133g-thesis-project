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

from typing import List, Optional

from ......node import Node
from .....mutation_context import MutationContext
from ....mutation_rule import MutationRule, MutationRecord
from .cstyle_for_loop_strategy import CstyleForLoopStrategy


class CppForLoopStrategy(CstyleForLoopStrategy):
    """
    Strategy for transforming C++ for-loops into equivalent while-loops.
    """

    def is_valid(self, node: Node) -> bool:
        """
        Checks whether the node represents a C++ for-loop.

        Args:
            node (Node): The CST node.

        Returns:
            bool: True if node is a C++ for-loop or enhanced for-loop.
        """
        return node.type in {
            "for_statement",
        }

    def apply(
        self, node: Node, rule: MutationRule, context: MutationContext, indent_unit: str
    ) -> List[MutationRecord]:
        """
        Transforms a traditional Java for-loop into a while-loop equivalent.

        Args:
            node (Node): The traditional for-loop node.
            rule (MutationRule): Rule instance used for recording mutations.
            context (MutationContext): Shared mutation context.
            indent_unit (str): The indentation unit for the language.

        Returns:
            List[MutationRecord]: A list of the performed mutations.
        """
        records = []
        print(f"Start:\n{node.to_code()}")
        print(f"Found valid node: {node.type}")
        print(f"Node text: {node.text}")
        print("Printing children for debugging:")
        for child in node.children:
            print(f" - {child.type}")
            print(f"   Text: {child.text}")
            if child.semantic_label:
                print(f"   Semantic label: {child.semantic_label}")
            for child_child in child.children:
                print(f"   - {child_child.type}: {child_child.text}")
                if child_child.semantic_label:
                    print(f"     Semantic label: {child_child.semantic_label}")
                print("   CChildren:")
                for cc in child_child.children:
                    print(f"     - {cc.type}: {cc.text}")
        for_node, init_node, condition_node, update_node, body = self._extract_for_loop_components(
            node
        )

        # If body is not block or contains no statements, skip transformation
        if body is None or not self._has_effective_body(body):
            return []

        opening_brace, closing_brace = self._get_block_braces(body)
        if opening_brace is None or closing_brace is None:
            return (
                []
            )  # If we can't find braces, we can't reliably insert updates, so skip transformation

        # Build source code strings for init and update sections
        init_source = self._emit_stmt(init_node.to_code()) if init_node else None
        update_source = self._emit_stmt(update_node.to_code()) if update_node else None

        # Delete init and update nodes from the for loop header and clean up remaining formatting tokens
        records.extend(self._delete_nodes([init_node], rule))
        records.extend(self._delete_nodes([update_node], rule))
        records.extend(self._clean_for_loop_header(node, rule))

        records.append(self._substitute(for_node, "while", "while", rule))

        # Insert a default "true" condition if the condition section is empty, to maintain loop semantics
        if not condition_node:
            records.append(self._insert_default_true_condition(node, context, rule))

        # Get what will be needed for initializer insertions:
        siblings = node.children
        for_node_idx = siblings.index(for_node)
        init_indent = self._get_indent(node) + indent_unit

        # Insert initializer before the transformed while loop
        if init_source:
            records.extend(
                self._insert_segments(
                    context,
                    node,
                    for_node.start_point,
                    [("whitespace", init_indent), ("newline", "\n"), ("initializer", init_source)],
                    for_node_idx,
                    rule,
                )
            )

        # Get what will be needed for update insertions:
        last_statement = self._find_last_statement(body)
        body_insert_idx = body.children.index(last_statement) + 1
        node_at_idx = body.children[body_insert_idx]
        body_indent = init_indent + indent_unit

        # Insert update at the end of the loop body
        if update_source:
            records.extend(
                self._insert_segments(
                    context,
                    body,
                    node_at_idx.start_point,
                    [
                        ("update_expression", update_source),
                        ("whitespace", body_indent),
                        ("newline", "\n"),
                    ],
                    body_insert_idx,
                    rule,
                )
            )

        # Add braces around the body to preserve variable scoping for initializers:

        # Insert opening brace for enclosing block
        records.extend(
            self._insert_segments(
                context,
                node,
                for_node.start_point,
                [("whitespace", init_indent), ("newline", "\n"), ("{", "{")],
                for_node_idx,
                rule,
            )
        )
        # Insert closing brace for enclosing block
        closing_brace_insert_idx = len(node.children)
        records.extend(
            self._insert_segments(
                context,
                node,
                node.end_point,
                [
                    ("}", "}"),
                    ("whitespace", init_indent.removesuffix(indent_unit)),
                    ("newline", "\n"),
                ],
                closing_brace_insert_idx,
                rule,
            )
        )
        # Add extra indentation to the body to account for the new block
        records.extend(self._apply_indent_reformat(node, indent_unit, rule))

        if opening_brace.start_point[0] == closing_brace.start_point[0]:

            # Handle opening brace: target is the node immediately after '{'
            opening_idx = body.children.index(opening_brace)
            self._ensure_newline_and_indent(
                context, body, opening_brace, opening_idx + 1, body_indent, rule, records
            )
            # Handle closing brace: target is the position immediately before '}'
            closing_idx = body.children.index(closing_brace)
            self._ensure_newline_and_indent(
                context, body, closing_brace, closing_idx, init_indent, rule, records
            )

        return records

    def _extract_for_loop_components(
        self, node: Node
    ) -> tuple[Optional[Node], Optional[Node], Optional[Node], Optional[Node], Optional[Node]]:
        """
        Extracts the structural components of a C++ for-loop node from a CST.

        This method traverses the direct children of the provided node and identifies
        the key components of a for-loop structure:
        initialization, condition, update expression, and loop body.

        Args:
            node (Node): The CST node expected to represent a for-loop structure.

        Returns:
            tuple:
                - for_node (Node | None): The identified 'for' node if present.
                - init_node (Node | None): Initialization expression node.
                - condition_node (Node | None): Loop condition expression node.
                - update_node (Node | None): Update expression node.
                - body (Node | None): The block node representing the loop body, if present.
        """

        init_node = None
        condition_node = None
        update_node = None
        body = None
        for_node = None

        section = 0  # 0 = init, 1 = condition, 2 = update

        for child in node.children:
            match child.type:

                case "for":
                    for_node = child

                case ";":
                    section += 1

                case "declaration":
                    init_node = child
                    section += 1  # The ; is inluded in the declarationo node, so we move to the next section immediately

                case (
                    "assignment_expression"
                    | "comma_expression"
                    | "binary_expression"
                    | "true"
                    | "false"
                    | "parenthesized_expression"
                    | "identifier"
                    | "call_expression"
                    | "update_expression"
                ):

                    if section == 0:
                        init_node = child
                    elif section == 1:
                        condition_node = child
                    elif section == 2:
                        update_node = child

                case "compound_statement":
                    body = child

                case _:
                    continue

        return for_node, init_node, condition_node, update_node, body
