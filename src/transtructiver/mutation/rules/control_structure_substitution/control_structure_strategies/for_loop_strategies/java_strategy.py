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
        for_node, init_nodes, condition_nodes, update_nodes, body = (
            self._extract_for_loop_components(node)
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

    def _force_block_break(self, context, body, anchor_node, indent, rule, after=True):
        """
        Adjusts the block structure to ensure a newline and proper indentation
        exist around a brace.

        This method checks for existing whitespace adjacent to an anchor node
        (typically a brace) and reformats it to match the desired indentation.
        It then inserts a newline to force the block onto a new line, effectively
        expanding one-liner blocks into a standard multi-line format.

        Args:
            context (MutationContext):
                The shared execution context used for unique ID generation.
            body (Node):
                The 'block' node representing the loop body being modified.
            anchor_node (Node):
                The specific child node (e.g., '{' or '}') to use as a
                positional anchor.
            indent (str):
                The whitespace string to be used for reformatting (e.g., the
                calculated indentation for the body or the loop header).
            rule (MutationRule):
                The rule instance used to record the reformat and insert operations.
            after (bool):
                If True, the newline is ensured after the anchor node (for '{').
                If False, it is ensured before the anchor node (for '}').
                Defaults to True.

        Returns:
            List[MutationRecord]:
                A list of records for the whitespace reformatting and the
                newline insertion.
        """
        idx = body.children.index(anchor_node)
        target_idx = idx + 1 if after else idx

        # Look for existing whitespace to reformat
        adj_idx = idx + 1 if after else idx - 1
        adj_node = body.children[adj_idx] if 0 <= adj_idx < len(body.children) else None

        records = []
        if adj_node and adj_node.type == "whitespace":
            records.append(rule.record_reformat(adj_node, indent))

        # Insert the newline node
        point = anchor_node.end_point if after else anchor_node.start_point
        records.append(self._insert_node(context, body, point, "newline", "\n", target_idx, rule))
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

    def _generate_iterator_name(self, context: MutationContext) -> str:
        """
        Generates a unique name for a new iterator identifier.

        Args:
            context (MutationContext): Shared context containing forbidden_names.

        Returns:
            str: A unique, claimed identifier string.
        """
        base = "iter"
        candidate = base
        counter = 0

        while candidate in context.forbidden_names:
            candidate = f"{base}_{counter}"
            counter += 1

        context.forbidden_names.add(candidate)
        return candidate
