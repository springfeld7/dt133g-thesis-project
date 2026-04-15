"""Java loop substitution strategy.

Handles both Java for loop forms:

- Traditional for-loops:

    for (init; condition; increment) { body }

Converted into:

    init;
    while (condition) {
        body
        increment
    }

- Enhanced for-loops (for-each):

    for (Type x : iterable) { body }

Converted into:

    Iterator-based while-loop equivalent:

    Iterator<Type> it = iterable.iterator();
    while (it.hasNext()) {
        Type x = it.next();
        body
    }
"""

from typing import List, Optional

from ......node import Node
from .....mutation_context import MutationContext
from ....mutation_rule import MutationRule, MutationRecord
from .base_for_loop_strategy import BaseForLoopStrategy


class JavaForLoopStrategy(BaseForLoopStrategy):
    """
    Strategy for transforming Java for-loops into equivalent while-loops.
    """

    _STRUCTURAL_IGNORED = {"{", "}"}
    _FORMATTING_IGNORED = {"whitespace", "newline"}
    _IGNORED_BLOCK_TOKENS = _STRUCTURAL_IGNORED | _FORMATTING_IGNORED

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
            "enhanced_for_statement",
        }

    def apply(
        self, node: Node, rule: MutationRule, context: MutationContext, indent_unit: str
    ) -> List[MutationRecord]:
        """
        Transforms a Java for-loop into a while-loop equivalent.

        Dispatches transformation based on loop type.

        Args:
            node (Node): The for-loop node.
            rule (MutationRule): Rule instance used for recording mutations.
            context (MutationContext): Shared mutation context.
            indent_unit (str): The indentation unit for the language.

        Returns:
            List[MutationRecord]: Replace operation transforming the loop.
        """
        if node.type == "for_statement":
            return self._transform_traditional(node, rule, context, indent_unit)

        if node.type == "enhanced_for_statement":
            return self._transform_enhanced(node, rule, context, indent_unit)

        return []

    def _transform_traditional(
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
        for_node, init_nodes, condition_nodes, update_nodes, body = (
            self._extract_for_loop_components(node)
        )

        # If body contains no statements, skip transformation
        if not self._has_effective_body(body):
            return []

        # Build source code strings for init and update sections
        init_sources = self._normalize_init_nodes(init_nodes)
        update_sources = [self._emit_stmt(n.to_code()) for n in update_nodes]

        # Delete init and update nodes from the for loop header and clean up remaining formatting tokens
        records.extend(self._delete_nodes(init_nodes, rule))
        records.extend(self._delete_nodes(update_nodes, rule))
        records.extend(self._clean_for_loop_header(node, rule))

        # Substitute the targeted nodes
        records.append(self._substitute(node, "while_statement", node.text, rule))
        records.append(self._substitute(for_node, "while", "while", rule))

        # Insert a default "true" condition if the condition section is empty, to maintain loop semantics
        if not condition_nodes:
            records.append(self._insert_default_true_condition(node, context, rule))

        # Check if we need extra braces to maintain scope of initializers
        needs_scope_block = any(n.type == "local_variable_declaration" for n in init_nodes)

        # Get what will be needed for initializer insertions:
        siblings = node.children
        for_node_idx = siblings.index(for_node)
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
        body_insert_idx = body.children.index(last_statement) + 1
        node_at_idx = body.children[body_insert_idx]
        body_indent = self._get_indent(last_statement)
        body_indent = body_indent + indent_unit if needs_scope_block else body_indent

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

        return records

    def _transform_enhanced(
        self, node: Node, rule: MutationRule, context: MutationContext, indent_unit: str
    ) -> List[MutationRecord]:
        """
        Transforms an enhanced Java for-loop (for-each) into an iterator-based while-loop equivalent.

        Args:
            node (Node): The enhanced for-loop node.
            rule (MutationRule): Rule instance used for recording mutations.
            context (MutationContext): Shared mutation context.
            indent_unit (str): The indentation unit for the language.

        Returns:
            List[MutationRecord]: A list of the performed mutations.
        """
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
        new_code = "fsd"
        return []

    def _extract_for_loop_components(
        self, node: Node
    ) -> tuple[Optional[Node], list[Node], list[Node], list[Node], Optional[Node]]:
        """
        Extracts the structural components of a for-loop node from a CST.

        This method traverses the direct children of the provided node and identifies
        the key components of a for-loop structure, separating them into initialization
        expressions, update expressions, and the loop body.

        Args:
            node (Node): The CST node expected to represent or contain a for-loop structure.

        Returns:
            tuple:
                - for_node (Node | None): The identified 'for' node if present.
                - init_nodes (list[Node]): A list of initialization-related nodes extracted
                from variable declarations and assignment expressions.
                - condition_nodes (list[Node]): A list of condition-expression nodes found in the loop.
                - update_nodes (list[Node]): A list of update-expression nodes found in the loop.
                - body (Node | None): The block node representing the loop body, if present.
        """
        init_nodes = []
        condition_nodes = []
        update_nodes = []
        body = None
        for_node = None

        section = 0  # 0 = init, 1 = condition, 2 = update

        for child in node.children:
            match child.type:

                case ";":
                    section += 1

                case "for":
                    for_node = child

                case "local_variable_declaration":
                    init_nodes.append(child)
                    section += 1  # In local_variable_declaration, the ; is included as a child

                case "assignment_expression":
                    if section == 0:
                        init_nodes.append(child)
                    elif section == 2:
                        update_nodes.append(child)

                case "binary_expression":
                    if section == 1:
                        condition_nodes.append(child)

                case "update_expression":
                    update_nodes.append(child)

                case "method_invocation":
                    if section == 0:
                        init_nodes.append(child)
                    elif section == 1:
                        condition_nodes.append(child)
                    elif section == 2:
                        update_nodes.append(child)

                case "block":
                    body = child

                case _:
                    continue

        return for_node, init_nodes, condition_nodes, update_nodes, body

    def _emit_stmt(self, text: str) -> str:
        """
        Emits a complete statement as source code.

        This helper ensures that all generated statements are properly
        terminated with a semicolon.

        Args:
            text (str): The raw statement text without termination.

        Returns:
            str: A fully formatted statement ending with ';'.
        """
        return f"{text};"

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

    def _substitute(
        self, node: Node, new_type: str, new_text: str, rule: MutationRule
    ) -> MutationRecord:
        """
        Mutates node type/text and returns the substitution record.

        Args:
            node: The node to mutate.
            new_type: The new type to assign to the node.
            new_text: The new text to assign to the node.
            rule: The mutation rule instance used for recording the substitution.

        Returns:
            The generated mutation record.
        """
        old_type = node.type
        node.type = new_type
        node.text = new_text
        return rule.record_substitute(node, old_type)

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

    def _insert_node(self, context, parent, point, type, text, index, rule):
        """
        Creates and inserts a Node, then returns the generated insertion record.

        Args:
            context: The execution context providing unique IDs.
            parent: The parent Node receiving the new child.
            point: The location where the new node will be inserted.
            type (str): The node type for the Node and the record.
            text (str): The text content for the Node and the record.
            index (int): The position to insert the child in the parent's list.
            records (list): The list where the record is stored.
            rule: The rule object used to generate the record.

        Returns:
            dict: The record object generated by rule.record_insert.
        """
        new_node = Node(
            start_point=(context.next_id(), -1),
            end_point=point,
            type=type,
            text=text,
        )
        new_node.parent = parent
        parent.children.insert(index, new_node)

        record = rule.record_insert(
            point=new_node.start_point,
            insertion_point=point,
            new_text=new_node.text,
            new_type=type,
        )

        return record

    def _insert_segments(self, context, parent, point, segments, index, rule):
        """
        Inserts a batch of nodes and returns the resulting records.

        Args:
            context: The execution context providing unique IDs.
            parent: The parent Node receiving the new children.
            point: The location where the new nodes will be inserted.
            segments (list[tuple]): A list of (type, text) tuples to insert.
            index (int): The starting position for insertions.
            rule: The rule object used to generate records.

        Returns:
            list: A list of record objects generated during insertion.
        """
        records = []
        for node_type, content in segments:
            records.append(
                self._insert_node(context, parent, point, node_type, content, index, rule)
            )

        return records

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
