"""cstyle_for_loop_strategy.py

Base strategy for transforming C-style 'for' loops into 'while' loops.
"""

from abc import abstractmethod
from typing import Optional

from .base_for_loop_strategy import BaseForLoopStrategy
from .....rules.mutation_rule import MutationRecord
from ......node import Node


class CstyleForLoopStrategy(BaseForLoopStrategy):
    """
    Strategy for transforming C-style 'for' loops into 'while' loops.
    This strategy applies to languages like Java and C++ that use C-style 'for' loops.
    """

    _STRUCTURAL_IGNORED = {"{", "}"}
    _FORMATTING_IGNORED = {"whitespace", "newline"}
    _IGNORED_BLOCK_TOKENS = _STRUCTURAL_IGNORED | _FORMATTING_IGNORED

    def is_effective_node(self, node: Node) -> bool:
        """
        Determines if a node is an effective statement, ignoring formatting and structural tokens.

        Args:
            node (Node): The node to check.

        Returns:
            bool: True if the node is an effective statement, False if it's only formatting or structural.
        """
        return node.type not in self._IGNORED_BLOCK_TOKENS

    def _has_effective_body(self, block: Node) -> bool:
        """
        Checks if the block contains any effective statements, ignoring whitespace and braces.

        Args:
            block (Node): The block node to check.

        Returns:
            bool: True if there are effective statements, False otherwise.
        """
        return any(self.is_effective_node(child) for child in block.children)

    def _find_last_statement(self, block: Node) -> Optional[Node]:
        """
        Finds the last effective statement in a block, ignoring whitespace and braces.

        Args:
            block (Node): The block node to check.

        Returns:
            Optional[Node]: The last effective statement, or None if not found.
        """
        return next(
            (child for child in reversed(block.children) if self.is_effective_node(child)),
            None,
        )

    def _emit_stmt(self, text: str) -> str:
        """
        Emits a complete statement as source code.

        This helper ensures that all generated statements are properly
        terminated with a semicolon.

        Args:
            text (str): The raw statement text without termination.

        Returns:
            str: A fully formatted statement ending with ';' (if not already present).
        """
        text = text.rstrip()

        if text.endswith(";"):
            return text

        return f"{text};"

    def _get_block_braces(self, body):
        """
        Retrieves the opening and closing brace nodes from a block body.

        This method scans the children of the given body node and returns the
        first encountered opening brace ("{") and closing brace ("}") nodes.

        Args:
            body: The AST node representing a block body whose children may include
                brace tokens.

        Returns:
            tuple: A tuple (opening_brace, closing_brace) where each element is either
                the corresponding brace node or None if not found.
        """
        opening = next((c for c in body.children if c.type == "{"), None)
        closing = next((c for c in body.children if c.type == "}"), None)
        return opening, closing

    def _clean_for_loop_header(self, node: Node, rule) -> list[MutationRecord]:
        """
        Removes syntactic noise inside a for-loop header and records deletions.

        This method scans the children of a for-loop node, tracks whether traversal
        is inside the parentheses, and collects non-semantic formatting nodes such
        as whitespace, commas, and semicolons that may remain after structural
        mutations.

        It generates and returns mutation records for the deletion of these nodes.

        Args:
            node (Node): The for-loop CST node whose header is being cleaned.
            rule: MutationRule used to generate deletion records.

        Returns:
            list[MutationRecord]: A list of mutation records for the deletions.
        """
        records = []
        inside = False
        nodes_to_delete = []

        for c in node.children:
            if c.type == "(":
                inside = True
            elif c.type == ")":
                inside = False
            elif inside and c.type in ("whitespace", ";", ","):
                nodes_to_delete.append(c)

        for n in nodes_to_delete:
            records.append(rule.record_delete(n.parent, n))

        return records

    def _ensure_newline_and_indent(
        self, context, body, anchor_node, target_idx, indent_value, rule, records
    ):
        """
        Ensures that a specific index in the body contains a newline and the correct indentation.
        If whitespace already exists at the target location, it reformats it; otherwise,
        it inserts new whitespace and newline nodes.

        Args:
            context (Context): The current transformation context.
            body (Node): The parent node containing the children to modify.
            anchor_node (Node): The node used as a positional reference for start_point.
            target_idx (int): The index in body.children where the nodes should be checked/inserted.
            indent_value (str): The string value of the indentation to apply.
            rule (Rule): The rule object responsible for recording changes.
            records (list): The list of transformation records to append to.

        Returns:
            None: Appends records in-place.
        """
        # Check if we are within bounds and if the node at target_idx is already whitespace
        if 0 <= target_idx < len(body.children) and body.children[target_idx].type == "whitespace":
            records.append(rule.record_reformat(body.children[target_idx], indent_value))
        else:
            records.append(
                self._insert_node(
                    context,
                    body,
                    anchor_node.start_point,
                    "whitespace",
                    indent_value,
                    target_idx,
                    rule,
                )
            )

        # Always insert the newline
        records.append(
            self._insert_node(
                context, body, anchor_node.start_point, "newline", "\n", target_idx, rule
            )
        )

    def _apply_indent_reformat(self, node, indent_unit, rule):
        """
        Recursively increases indentation for all start-of-line whitespace.

        Args:
            context: Execution context for unique IDs.
            node (Node): The node to begin traversal from.
            indent_unit (str): The indentation string to append.
            rule: The rule object providing record_reformat.

        Returns:
            List[MutationRecord]: All reformat records generated.
        """
        records = []
        for n in node.traverse():
            # Condition: It's a whitespace node AND it starts at column 0
            if n.type == "whitespace" and n.start_point[1] == 0:
                # Append indent_unit to the existing whitespace text
                records.append(rule.record_reformat(n, n.text + indent_unit))

        return records

    def _insert_default_true_condition(self, node, context, rule) -> MutationRecord:
        """
        Inserts a default 'true' condition if the condition section is empty.

        Args:
            node: The CST node representing the for-loop (or condition container).
            context: Context object providing utilities such as ID generation.
            rule: Rule object used to create mutation records.

        Returns:
            MutationRecord: The mutation record produced by this transformation.
        """
        idx, closing_node = next((i, c) for i, c in enumerate(node.children) if c.type == ")")

        true_node = Node(
            start_point=(context.next_id(), -1),
            end_point=closing_node.start_point,
            type="binary_expression",
            text="true",
        )

        true_node.parent = node
        node.children.insert(idx, true_node)

        return rule.record_insert(
            point=true_node.start_point,
            insertion_point=closing_node.start_point,
            new_text=true_node.text,
            new_type=true_node.type,
        )
