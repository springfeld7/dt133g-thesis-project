"""C++ semantic annotator."""

from ....node import Node
from .base_annotator import BaseAnnotator


class CppAnnotator(BaseAnnotator):
    language = "cpp"
    direct_type_labels: dict[str, str] = {
        "preproc_include": "import_scope",
        "namespace_definition": "namespace_scope",
        "class_specifier": "class_scope",
        "struct_specifier": "class_scope",
        "enum_specifier": "class_scope",
        "function_definition": "function_scope",
        "lambda_expression": "function_scope",
        "for_statement": "loop_scope",
        "while_statement": "loop_scope",
        "for_range_loop": "loop_scope",
        "declaration": "declaration_scope",
        "field_declaration": "declaration_scope",
        "assignment_expression": "assignment_scope",
        "namespace_identifier": "namespace_name",
        "if_statement": "condition_scope",
        "call_expression": "call_scope",
        "binary_expression": "operation_scope",
        "return_statement": "return_scope",
    }
    parent_type_labels: dict[str, str] = {
        "class_specifier": "class_name",
        "struct_specifier": "class_name",
        "enum_specifier": "class_name",
        "destructor_name": "class_name",
        "function_declarator": "function_name",
        "call_expression": "function_name",
        "parameter_declaration": "parameter_name",
        "optional_parameter_declaration": "parameter_name",
        "argument_list": "argument_name",
        "type_alias_declaration": "type_name",
        "throw_statement": "exception_name",
    }

    def handle_special_node(self, node: Node, parent: Node) -> bool:
        if node.type == "comment" and node.text is not None:
            if node.text.startswith("//"):
                node.semantic_label = "line_comment"
                return True
            if node.text.startswith("/*"):
                node.semantic_label = "block_comment"
                return True

        return False

    def handle_special_identifier(self, node: Node, parent: Node) -> str | None:
        if node.type == "field_identifier" and parent.type == "field_expression":
            return "function_name" if parent.field == "function" else "property_name"
        return None
