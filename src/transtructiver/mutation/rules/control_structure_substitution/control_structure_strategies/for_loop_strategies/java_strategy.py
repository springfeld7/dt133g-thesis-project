"""Java loop substitution strategy.

Handles traditional Java for loops:

    for (init; condition; update) { body; }

Converted into:

    init;
    while (condition) {
        body;
        update;
    }

If init contains variable declarations, extra braces are added around the body to preserve variable scoping:

    {
        init;
        while (condition) {
            body;
            update;
        }
    }
"""

from operator import indexOf
from typing import List, Optional

from .cstyle_for_loop_strategy import CstyleForLoopStrategy
from ......node import Node
from .....mutation_context import MutationContext
from ....mutation_rule import MutationRule, MutationRecord


class JavaForLoopStrategy(CstyleForLoopStrategy):
    """
    Strategy for transforming Java for-loops into equivalent while-loops.
    """

    def is_valid(self, node: Node) -> bool:
        """
        Checks whether the node represents a Java for-loop.

        Args:
            node (Node): The CST node.

        Returns:
            bool: True if node is a Java for-loop or enhanced for-loop.
        """
        return node.type in {
            "for_statement",
        }

    def apply(
        self, node: Node, rule: MutationRule, context: MutationContext, indent_unit: str, level: int
    ) -> List[MutationRecord]:
        """
        Transforms a traditional Java for-loop into a while-loop equivalent.

        Args:
            node (Node): The traditional for-loop node.
            rule (MutationRule): Rule instance used for recording mutations.
            context (MutationContext): Shared mutation context.
            indent_unit (str): The indentation unit for the language.
            level (int): The transformation level to apply. (Unused for Java)

        Returns:
            List[MutationRecord]: A list of the performed mutations.
        """
        records = []
        for_node, init_nodes, condition_nodes, update_nodes, body = (
            self._extract_for_loop_components(node)
        )

        # If no for_node or body or contains no statements, skip transformation
        if for_node is None or body is None or not self._has_effective_body(body):
            return []

        opening_brace, closing_brace = self._get_block_braces(body)
        if opening_brace is None or closing_brace is None:
            return (
                []
            )  # If we can't find braces, we can't reliably insert updates, so skip transformation

        # Build source code strings for init and update sections
        init_sources = self._normalize_init_nodes(init_nodes)
        update_sources = [self._emit_stmt(n.to_code()) for n in update_nodes]

        # Delete init and update nodes from the for loop header and clean up remaining formatting tokens
        records.extend(self._delete_nodes(init_nodes, rule))
        records.extend(self._delete_nodes(update_nodes, rule))
        records.extend(self._clean_for_loop_header(node, rule))

        records.append(self._substitute(for_node, "while", "while", rule))

        # Insert a default "true" condition if the condition section is empty, to maintain loop semantics
        if not condition_nodes:
            records.append(self._insert_default_true_condition(node, context, rule))

        # Check if we need extra braces to maintain scope of initializers
        needs_scope_block = any(n.type == "local_variable_declaration" for n in init_nodes)

        # Get what will be needed for initializer insertions:
        siblings = node.children
        for_node_idx = indexOf(siblings, for_node)
        init_indent = self._get_indent(node)
        init_indent = init_indent + indent_unit if needs_scope_block else init_indent

        # Insert initializers before the transformed while loop
        for src in reversed(init_sources):
            records.extend(
                self._insert_segments(
                    context,
                    node,
                    for_node.start_point,
                    [("whitespace", init_indent), ("newline", "\n"), ("initializer", src)],
                    for_node_idx,
                    rule,
                )
            )

        # Get what will be needed for update insertions:
        last_statement = self._find_last_statement(body)
        body_insert_idx = indexOf(body.children, last_statement) + 1
        node_at_idx = body.children[body_insert_idx]
        body_indent = init_indent + indent_unit

        # Insert updates at the end of the loop body
        for src in reversed(update_sources):
            records.extend(
                self._insert_segments(
                    context,
                    body,
                    node_at_idx.start_point,
                    [("update_expression", src), ("whitespace", body_indent), ("newline", "\n")],
                    body_insert_idx,
                    rule,
                )
            )

        if needs_scope_block:
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
    ) -> tuple[Optional[Node], list[Node], list[Node], list[Node], Optional[Node]]:
        """
        Extracts the structural components of a for-loop node from a CST.

        Args:
            node (Node): The CST node expected to represent or contain a for-loop structure.

        Returns:
            tuple:
                - for_node (Node | None): The identified 'for' node if present.
                - init_nodes (list[Node]): A list of initialization-related nodes.
                - condition_nodes (list[Node]): A list of condition-expression nodes.
                - update_nodes (list[Node]): A list of update-expression nodes.
                - body (Node | None): The block node representing the loop body.
        """
        init_nodes, condition_nodes, update_nodes = [], [], []
        for_node, body = None, None
        section = 0  # 0 = init, 1 = condition, 2 = update

        # Map sections to their respective storage lists
        sections = {0: init_nodes, 1: condition_nodes, 2: update_nodes}

        for child in node.children:
            match child.type:
                case ";":
                    section += 1
                case "for":
                    for_node = child
                case "block":
                    body = child
                case "local_variable_declaration":
                    init_nodes.append(child)
                    section += 1  # Semicolon is internal to this node type
                case "update_expression":
                    update_nodes.append(child)
                case (
                    "assignment_expression"
                    | "method_invocation"
                    | "binary_expression"
                    | "identifier"
                    | "parenthesized_expression"
                    | "true"
                    | "false"
                ):
                    # Only append if the current section is valid for this node type
                    if section in sections:
                        sections[section].append(child)

        return for_node, init_nodes, condition_nodes, update_nodes, body

    def _normalize_init_nodes(self, init_nodes: list[Node]) -> list[str]:
        """
        Normalizes initialization nodes from a for-loop into valid source statements.

        This method converts CST nodes representing initialization logic into properly
        formatted statements. It handles multiple node types:

        - local_variable_declaration:
            Split into individual variable declarators and emit one statement per variable.
            Example: `int i = 0, j = 2;` → `int i = 0;`, `int j = 2;`

        - assignment_expression:
            Emitted directly as a single statement.

        - method_invocation:
            Emitted directly as a single statement.

        All emitted statements are passed through `_emit_stmt` to ensure consistent
        termination with semicolons and newlines.

        Args:
            init_nodes (list[Node]): List of CST nodes representing initialization
                components of a for-loop.

        Returns:
            list[str]: A list of normalized initialization statements as source code strings.
        """
        init_sources = []

        for node in init_nodes:

            if node.type == "local_variable_declaration":
                base_type = ""

                for c in node.children:
                    if c.semantic_label == "type_name":
                        base_type = c.to_code()

                    elif c.type == "variable_declarator":
                        init_sources.append(self._emit_stmt(f"{base_type} {c.to_code()}"))

            elif node.type in {"assignment_expression", "method_invocation"}:
                init_sources.append(self._emit_stmt(node.to_code()))

        return init_sources
