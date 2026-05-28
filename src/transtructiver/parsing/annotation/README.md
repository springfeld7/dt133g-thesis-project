# Annotation system

This package provides semantic annotation for CST nodes so mutation and verification logic can work with normalized labels across languages such as Python, Java, and C++.

## Current structure

The main entry point is [annotator.py](annotator.py). It dispatches to the language-specific annotator classes in [annotators/](annotators) based on `root.language`.

The package currently includes:

- [annotator.py](annotator.py): dispatcher that selects the annotator for the active language
- [annotators/base_annotator.py](annotators/base_annotator.py): shared traversal and labeling logic
- [annotators/python_annotator.py](annotators/python_annotator.py): Python-specific labels
- [annotators/java_annotator.py](annotators/java_annotator.py): Java-specific labels
- [annotators/cpp_annotator.py](annotators/cpp_annotator.py): C++-specific labels
- [builtin_checker.py](builtin_checker.py): builtin keyword/profile matching
- [annotation_utils.py](annotation_utils.py): small helper predicates used by annotators

## Builtin profiling and language profiles

The builtin checker loads language-specific builtin keyword lists from [profiles/](profiles) and uses them to mark builtin names so they can be excluded from transformations such as renaming.

The currently bundled profiles are:

- `profiles/python_builtins.json`: Python built-in functions and types
- `profiles/java_builtins.json`: Java standard library classes and packages
- `profiles/cpp_builtins.json`: C++ STL types and functions

The checker compares a node's text against builtin names from the active language profile. If a match is found, the annotator marks the node as builtin.

## Profile generation helper

The file [profiles/build_profile.py](profiles/build_profile.py) is a maintainer utility for generating lightweight builtin profile JSON files from DevDocs data. It is not part of the runtime annotation pipeline.

Because DevDocs index formats can vary by language and version, the target DevDocs `index.json` must be manually inspected before parsing so the entry and type fields are interpreted correctly for the selected language.

## Extending with a new language

1. Create a new annotator class in [annotators/](annotators) by subclassing [base_annotator.py](annotators/base_annotator.py).
2. Set the class `language` attribute so the dispatcher can register it automatically.
3. Define any language-specific `direct_type_labels`, `direct_field_labels`, and `parent_type_labels` needed for your CST.
4. Use the helpers in [annotation_utils.py](annotation_utils.py) for common predicates such as named-child filtering and scope detection.
5. Update or add builtin profiles under [profiles/](profiles) if the language has builtin names that should be protected.

### Example skeleton

```python
from .base_annotator import BaseAnnotator


class MyLangAnnotator(BaseAnnotator):
    language = "mylang"
    direct_type_labels: Mapping[str, str] = {
        "class_definition": "class_scope",
        "function_definition": "function_scope",
    }
    parent_type_labels: Mapping[str, str] = {
        "class_definition": "class_name",
        "function_definition": "function_name",
    }

    def handle_special_node(self, node: Node, parent: Node) -> bool:
        """Handle rare syntax-specific node cases."""
        return False

    def handle_special_identifier(self, node: Node, parent: Node) -> str | None:
        """Handle rare identifier-specific cases before the shared maps."""
        return None

```

## Usage

Call `annotator.annotate(root)` after `root.language` has been set. The annotator updates `node.semantic_label` values in place and also resolves basic declaration/context information for identifiers.

---