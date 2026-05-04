"""Java semantic annotator."""

from typing import Mapping

from ....node import Node
from .base_annotator import BaseAnnotator


class JavaAnnotator(BaseAnnotator):
    language = "java"
    direct_type_labels: Mapping[str, str] = {
        "import_declaration": "import_scope",
        "class_declaration": "class_scope",
        "interface_declaration": "class_scope",
        "record_declaration": "class_scope",
        "enum_declaration": "class_scope",
        "annotation_type_declaration": "class_scope",
        "method_declaration": "function_scope",
        "lambda_expression": "function_scope",
        "for_statement": "loop_scope",
        "while_statement": "loop_scope",
        "enhanced_for_statement": "loop_scope",
        "local_variable_declaration": "declaration_scope",
        "field_declaration": "declaration_scope",
        "assignment_expression": "assignment_scope",
        "if_statement": "condition_scope",
        "method_invocation": "call_scope",
        "binary_expression": "operation_scope",
        "return_statement": "return_scope",
        "line_comment": "line_comment",
        "block_comment": "block_comment",
    }
    parent_type_labels: Mapping[str, str] = {
        "class_declaration": "class_name",
        "constructor_declaration": "class_name",
        "compact_constructor_declaration": "class_name",
        "record_declaration": "class_name",
        "interface_declaration": "class_name",
        "annotation_type_declaration": "class_name",
        "enum_declaration": "class_name",
        "method_declaration": "function_name",
        "annotation_type_element_declaration": "function_name",
        "formal_parameter": "parameter_name",
        "argument_list": "argument_name",
        "throw_statement": "exception_name",
    }

    def handle_special_identifier(self, node: Node, parent: Node) -> str | None:
        if parent.type == "method_reference" and _is_right_side_of_operator(node, parent, "::"):
            return "function_name"

        if parent.type == "method_invocation":
            if node.field == "name":
                return "function_name"
        if parent.type == "field_access":
            if node.field == "field":
                return "property_name"

        return None


def _is_right_side_of_operator(node: Node, parent: Node, operator: str) -> bool:
    found_operator = False
    for child in parent.children:
        if child.text == operator:
            found_operator = True
        elif found_operator and child == node:
            return True
    return False
