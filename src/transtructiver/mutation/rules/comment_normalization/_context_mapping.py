"""
Context mapping and replacement heuristics for comment normalization.

This module provides internal logic for the CommentNormalizationRule, including
context-sensitive templates and operator mappings used to generate or normalize
code comments based on AST/CST node context. It is not intended for public use
outside the comment-normalization rule itself.

Main contents:
- Context-aware comment templates (short, medium, long)
- Operator-to-description mappings for multiple languages
- Helper functions for extracting context from annotated nodes
- Internal logic for selecting and formatting replacement comments

This module is private to the comment normalization rule and may change without notice.
"""

import re

from ....node import Node


_SHORT_TEMPLATES = {
    # Short, context-aware comment templates for line/block comments.
    "line": {
        "class": "Class {context}.",
        "function": "Execute {context}.",
        "assignment": "Set {context}.",
        "operation": "Do {context}.",
        "loop": "Iterate {context}.",
        "condition": "Check {context}.",
        "alternative": "Else branch.",
        "return": "Return {context}.",
        "call": "Invoke {context}.",
        "outcommented": "Logic placeholder.",
        "todo": "Pending task.",
    },
    "block": {
        "module": "The {context} module.",
        "class": "The {context} definition.",
        "function": "The {context} routine.",
        "assignment": "Update {context}.",
        "operation": "Perform {context}.",
        "loop": "Traverse {context}.",
        "condition": "The {context} validation.",
        "alternative": "Fallback block.",
        "return": "Output {context}.",
        "call": "{context} invocation.",
        "outcommented": "Implementation deferred.",
        "todo": "Action required.",
    },
}


_MEDIUM_TEMPLATES = {
    # Medium-length, context-aware comment templates for line/block comments.
    "line": {
        "class": "Definition for the {context}.",
        "function": "Handles the execution of the {context} routine.",
        "assignment": "Updates the value assigned to {context}.",
        "operation": "Evaluates the logic for the {context} expression.",
        "loop": "Loops {context}.",
        "condition": "Validates the {context} conditional state requirements.",
        "alternative": "Handles the alternative logic for this block.",
        "return": "Produces the resulting {context} output value.",
        "call": "Calls the {context} external functional interface.",
        "outcommented": "Supplemental code following the primary return.",
        "todo": "Outstanding item requiring future technical attention.",
    },
    "block": {
        "module": "Documentation for the {context} module implementation.",
        "class": "Represents the structural model for the {context}.",
        "function": "Implementation of the {context} functional logic.",
        "assignment": "Sets the underlying data for the {context} reference.",
        "operation": "Performs evaluation of the {context} expression.",
        "loop": "Iteration logic repeated {context}.",
        "condition": "Evaluates the {context} logical conditional state.",
        "alternative": "Supplemental logic for the alternative branch.",
        "return": "Returns the resulting {context} data object.",
        "call": "Executes the request for the {context} procedure.",
        "outcommented": "Reserved for future logic following this block.",
        "todo": "Documentation of an outstanding technical requirement.",
    },
}


_LONG_TEMPLATES = {
    # Long, context-aware comment templates for line/block comments.
    "line": {
        "class": "Implementation of the primary logic and data model for the {context}.",
        "function": "Facilitates the functional logic and operational flow for the {context}.",
        "assignment": "Manages the internal state by assigning a new reference to the {context} identifier.",
        "operation": "Performs the binary operation and computes the result involving the {context} operands.",
        "loop": "Traverses {context} to perform repetitive processing on the data.",
        "condition": "Evaluates the logical condition to determine the branch for the {context} execution.",
        "alternative": "Executes the fallback procedure if the primary conditional requirements are not met.",
        "return": "Concludes the current execution scope and returns the calculated {context} reference.",
        "call": "Dispatches a request to the {context} procedure to perform a specific operation.",
        "outcommented": "This section is reserved for additional logic that is currently excluded from the main flow.",
        "todo": "Reference to an outstanding functional requirement to be addressed in a subsequent iteration.",
    },
    "block": {
        "module": "Provides the comprehensive logic, structural definitions,"
        " and interface requirements for the {context} component.",
        "class": "Defines the primary object model and internal logic associated with the {context}.",
        "function": "Manages the execution flow and internal state transitions for the {context}.",
        "assignment": "Modifies the internal representation and data state"
        + " associated with the {context} member variable.",
        "operation": "Evaluates the arithmetic or logical computation involving"
        + " the {context} operands and resulting values.",
        "loop": "Executes a repetitive cycle {context} to facilitate data processing and transformation.",
        "condition": "Checks the specific requirements for the {context} condition"
        + " to determine the appropriate execution branch.",
        "alternative": "Defines the execution path and associated logic when the"
        + " primary condition is evaluated as false.",
        "return": "Terminates the current functional context and returns the final"
        + " reference for the {context} value.",
        "call": "Dispatches an execution signal to the {context} interface to perform the required functional task.",
        "outcommented": "Supplemental logic segment maintained for reference but"
        + " excluded from the current execution path.",
        "todo": "Reference to a functional implementation or refactoring task that"
        + " must be resolved in a future version.",
    },
}


_OPERATORS = {
    # Maps common operators (arithmetic, logical, bitwise, assignment, etc.)
    # to human-readable descriptions for use in comment generation.
    # Arithmetic
    "+": "addition operation",
    "-": "subtraction operation",
    "*": "multiplication operation",
    "/": "division operation",
    "%": "modulus operation",
    "**": "exponentiation operation",  # Python
    "//": "floor division operation",  # Python
    # Comparison
    "==": "equality check",
    "!=": "inequality check",
    "<": "less-than check",
    ">": "greater-than check",
    "<=": "less-than-equal check",
    ">=": "greater-than-equal check",
    # Logical
    "&&": "logical and check",  # Java/C++
    "||": "logical or check",  # Java/C++
    "!": "logical negation check",  # Java/C++
    "and": "logical and check",  # Python
    "or": "logical or check",  # Python
    "not": "logical negation check",  # Python
    # Bitwise
    "&": "bitwise and check",
    "|": "bitwise or check",
    "^": "bitwise xor check",
    "~": "bitwise inversion check",
    "<<": "left shift check",
    ">>": "right shift check",
    ">>>": "unsigned right shift check",  # Java
    # Assignment (Compound)
    "+=": "addition assignment",
    "-=": "subtraction assignment",
    "*=": "multiplication assignment",
    "/=": "division assignment",
    "%=": "modulus assignment",
    "&=": "bitwise and assignment",
    "|=": "bitwise or assignment",
    "^=": "bitwise xor assignment",
    "<<=": "left shift assignment",
    ">>=": "right shift assignment",
    # Unary / Increment (Java/C++)
    "++": "increment operation",
    "--": "decrement operation",
}


def _get_comment_template(label: str, words: int) -> dict[str, str]:
    """
    Selects the appropriate comment template dictionary (short, medium, long)
    based on the number of words in the comment and the node type.

    Args:
        type: The node type (e.g., 'line_comment', 'block_comment').
        words: The number of words in the comment.

    Returns:
        A dictionary mapping semantic labels to template strings.
    """
    comment_type = label.split("_")[0]
    if words <= 3:
        return _SHORT_TEMPLATES[comment_type]
    if words <= 8:
        return _MEDIUM_TEMPLATES[comment_type]
    else:
        return _LONG_TEMPLATES[comment_type]


def _get_values_for_context(node: Node) -> list[str]:
    """
    Extracts a list of contextually relevant text values from a node's children.
    Filters out whitespace, comments, and type nodes, and includes operator/operand fields.

    Args:
        node: The AST/CST node to extract context from.

    Returns:
        A list of string values representing the context for comment generation.
    """
    text: list[str] = []

    def _is_valid(n: Node, row: int):
        # Determines if a child node is relevant for context extraction.
        return bool(
            n.end_point[0] == row
            and (
                (
                    n.text != n.type
                    and n.type not in ["whitespace", "newline"]
                    and not _label_contains(n, "comment")
                    and not _label_contains(n, "type")
                )
                or (n.field and any(l in n.field for l in ["operator", "left", "right"]))
            )
        )

    for n in node.traverse():
        if n.text and _is_valid(n, node.start_point[0]):
            text.append(n.text)

    if len(text) < 1:
        text.append("")
    return text


def _get_loop_or_condition_context(node: Node, words: int) -> str:
    """
    Generates a context string for loop or condition nodes, describing the
    collection or condition being iterated or checked.

    Args:
        node: The loop or condition node.
        words: The number of words in the comment (for verbosity).

    Returns:
        A string describing the loop/condition context for comment templates.
    """
    for child in node.traverse():
        if child.field in ["right", "value"]:
            if not child.semantic_label or not _label_contains(child, "name"):
                if len(child.children) > 0:
                    context_values = _get_values_for_context(child)

                    if len(context_values) < 4:
                        values = ", ".join(context_values)
                    else:
                        values = f'{", ".join(context_values[:3])}, ...'

                    return f"through [{values}]"
            elif _label_contains(child, "name"):
                collection = child.text if child.text else ""
                extra = "all values of "

                return f"through {extra if words > 3 else ''}{collection}"
        elif child.field in ["condition", "alternative"]:
            prefix = "if" if _label_contains(node, "condition_scope") else "while"

            return f'{prefix} ({" ".join(_get_values_for_context(child))})'

        return f'{"".join(_get_values_for_context(child)[0])}'

    return ""


def _get_class_or_function_context(node: Node, words: int) -> str:
    """
    Generates a context string for class or function declaration nodes.

    Args:
        node: The class or function node.
        words: The number of words in the comment (for verbosity).

    Returns:
        A string describing the class/function context for comment templates.
    """
    context_values: list[str] = _get_values_for_context(node)
    value_count: int = len(context_values)

    if _label_contains(node, "class"):
        if value_count > 1:
            suffix = ": " if value_count > 2 else " "
            super_class = f' component with implementation derived from{suffix}{", ".join(context_values[1:])}'
            return f"{context_values[0]}{super_class if words > 8 else ''}"

    if _label_contains(node, "function"):
        if value_count > 1:
            if node.semantic_label:
                suffix: str = "s: " if value_count > 2 else " "
                w = "parameter" if _label_contains(node, "name") else "argument"
                args = f' function call, using the {w}{suffix}{", ".join(context_values[1:])}'
                return f"{context_values[0]}{args if words > 8 else ''}"

    return f'{"".join(context_values[0])}'


def _get_label_and_context(node: Node, words: int) -> tuple[str, str]:
    """
    Determines the semantic label and context string for a node, used to select
    the appropriate comment template and fill its context.

    Args:
        node: The node to analyze.
        words: The number of words in the comment (for verbosity).

    Returns:
        (label, context) tuple for template selection and formatting.
    """
    context_values: list[str] = _get_values_for_context(node)
    operators = []
    label = ""

    for w in context_values:
        if w in _OPERATORS:
            operators.append(_OPERATORS[w])

    for c in node.traverse():
        if c.semantic_label:
            label = _get_label(c).split("_")[0]
            if len(operators) > 0 and label == "assignment":
                label = "operation"
            break

    if label == "operation":
        if words <= 8:
            return label, f'{", ".join(operators)}'
        else:
            operands = [w for w in context_values if w not in _OPERATORS]
            j = ", "
            return label, f"'{j.join(operands)}'"

    return label, " ".join(_get_values_for_context(node)[:1])


def _label_contains(node: Node, text: str) -> bool:
    """
    Checks if a node's semantic label contains a given substring.

    Args:
        node: The node to check.
        text: The substring to search for in the semantic label.

    Returns:
        True if the label contains the substring, False otherwise.
    """
    return bool(node.semantic_label and text in node.semantic_label)


def _get_label(node: Node) -> str:
    """
    Returns the semantic label of a node, or an empty string if not present.
    """
    return node.semantic_label if node.semantic_label else ""


def _is_loop_or_condition(node: Node) -> bool:
    """
    Returns True if the node represents a loop or condition (semantic label).
    """
    return any(_label_contains(node, l) for l in ["loop", "condition"])


def _is_declaration(node: Node) -> bool:
    """
    Returns True if the node represents a class or function declaration.
    """
    return any(_label_contains(node, l) for l in ["class", "function"])


def _non_space_children(node: Node) -> list[Node]:
    """
    Returns a list of child nodes that are not whitespace or newline.
    """
    return [n for n in node.children if not n.type in ["whitespace", "newline"]]


def _is_after_terminal(start_col: int, ancestor: Node, row: int) -> bool:
    """
    Checks if a node appears after a terminal statement (return, break, etc.)
    in the ancestor node at a given row.
    """
    for n in ancestor.traverse():
        if n.end_point[0] <= row and not start_col < n.start_point[1]:
            if not _label_contains(n, "comment") and n.type in [
                "return_statement",
                "break_statement",
                "continue_statement",
                "pass_statement",
            ]:
                return True
    return False


def _get_context_from_row(node: Node, row: int, ancestor: Node, words: int) -> tuple[str, str]:
    """
    Determines the semantic label and context for a node based on its position
    within its ancestor and the given row, for use in comment replacement.

    Args:
        node: The node being analyzed.
        row: The row to search for context.
        ancestor: The ancestor node providing context.
        words: The number of words in the comment (for verbosity).

    Returns:
        (label, context) tuple for template selection and formatting.
    """
    same_line_search = node.start_point[0] == row
    label = ""
    context: str = ""

    for child in ancestor.traverse():
        child_row = child.start_point[0]
        if child != node and child in _non_space_children(ancestor) and child_row == row:
            if _is_loop_or_condition(ancestor):
                label = _get_label(ancestor)
                context = _get_loop_or_condition_context(ancestor, words)
                break

            if _is_declaration(ancestor):
                context = _get_class_or_function_context(ancestor, words)
                label = _get_label(ancestor)
                break

            if _is_loop_or_condition(child):
                if same_line_search and any(
                    c.field == "alternative" for c in _non_space_children(child)
                ):
                    label = "aternative"
                else:
                    label = _get_label(child)
                context = _get_loop_or_condition_context(child, words)
                break

            if _is_declaration(child):
                context = _get_class_or_function_context(child, words)
                label = _get_label(child)
                break

            label, context = _get_label_and_context(child, words)
            if label == "" and context == "":
                if not same_line_search and _is_after_terminal(
                    node.start_point[1], ancestor, row - 2
                ):
                    label = "outcommented"
                    end = ";" if node.text and node.text.startswith(("//", "/*")) else ""
                    context = f"{end}"
                    break
                else:
                    label = "todo"
            break

    return label.split("_")[0], context


def _replace_context_mapping(node: Node, ancestor: Node) -> str:
    """
    Main entry point for context-based comment replacement.
    Given a comment node and its ancestor, selects and formats a replacement
    comment string using context-aware templates and extracted context values.

    Args:
        node: The comment node to replace.
        ancestor: The ancestor node providing context.

    Returns:
        A formatted replacement comment string.
    """
    # Determine if the comment is on the same row as another node.
    start_row = node.start_point[0]
    end_row = node.end_point[0]
    on_same_row = 0

    for n in _non_space_children(ancestor):
        if n.start_point[0] > end_row:
            break
        elif n != node and n.end_point[0] == start_row:
            on_same_row += 1

    search_row: int = start_row if on_same_row > 0 else end_row + 1
    words: int = len(re.findall(r"\w+", node.text)) if node.text else 0

    comment_template = (
        _get_comment_template(node.semantic_label, words)
        if node.semantic_label
        else _MEDIUM_TEMPLATES["line"]
    )
    template = ""

    label, context = _get_context_from_row(node, search_row, ancestor, words)
    if label in comment_template:
        template = comment_template[label]
    else:
        template = comment_template["todo"]

    return template.format(context=context)
