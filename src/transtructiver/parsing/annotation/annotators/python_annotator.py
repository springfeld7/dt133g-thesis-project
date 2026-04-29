"""Python semantic annotator."""

from ....node import Node
from .base_annotator import BaseAnnotator


class PythonAnnotator(BaseAnnotator):
    language = "python"
    direct_type_labels: dict[str, str] = {
        "import_statement": "import_scope",
        "import_from_statement": "import_scope",
        "class_definition": "class_scope",
        "function_definition": "function_scope",
        "lambda": "function_scope",
        "for_statement": "loop_scope",
        "while_statement": "loop_scope",
        "expression_statement": "declaration_scope",
        "assignment": "assignment_scope",
        "if_statement": "condition_scope",
        "elif_clause": "condition_scope",
        "call": "call_scope",
        "binary_operator": "operation_scope",
        "parameters": "parameter_scope",
        "lambda_parameters": "parameter_scope",
        "return_statement": "return_scope",
        "comment": "line_comment",
    }
    parent_type_labels: dict[str, str] = {
        "class_definition": "class_name",
        "function_definition": "function_name",
        "parameters": "parameter_name",
        "typed_parameter": "parameter_name",
        "default_parameter": "parameter_name",
        "typed_default_parameter": "parameter_name",
        "raise_statement": "exception_name",
        "type": "type_name",
    }

    def handle_special_node(self, node: Node, _parent: Node) -> bool:
        if node.type == "string":
            if any(
                child.type in {"string_start", "string_end"} and child.text in ('"""', "'''")
                for child in node.children
            ):
                node.semantic_label = "block_comment"
                return True

        return False

    def handle_special_identifier(self, node: Node, parent: Node) -> str | None:
        if parent.type == "attribute":
            if node.field == "attribute":
                return "function_name" if parent.field == "function" else "property_name"

        if parent.type == "argument_list":
            return "class_name" if parent.field == "superclasses" else "argument_name"

        if parent.type == "keyword_argument" and node.field == "name":
            return "parameter_name"

        return None
