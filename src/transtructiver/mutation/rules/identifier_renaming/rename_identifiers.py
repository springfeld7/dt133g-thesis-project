"""rename_identifiers.py

Identifier renaming mutation rule.

This module defines RenameIdentifiersRule, a concrete MutationRule that applies
deterministic identifier renaming over annotated CST nodes and emits
MutationRecord entries for each rename action.
"""

from ._rename_appendage import _build_appendage_name
from ....node import Node
from ..mutation_rule import MutationRecord, MutationRule
from ...mutation_types import MutationAction
from ....parsing.annotation.annotator import ROOT_TO_LANGUAGE


class RenameIdentifiersRule(MutationRule):
    """Rename eligible identifier nodes using a generated naming scheme.

    The rule targets semantic identifier categories (variables, parameters,
    properties, functions, classes), applies language-aware suffix formatting,
    and preserves scope consistency for declarations and references.
    """

    def __init__(self, level: int = 0, targets: list[str] | None = None) -> None:
        super().__init__()
        self.level = level
        self.targets = targets or []
        self.allowed_labels = self._resolve_target_labels(self.targets)
        self.scope: list[dict[str, str]] = []
        self.language: str = ""

    _RENAME_LEVEL = {
        0: _build_appendage_name,
    }

    _TARGET_TO_LABEL = {
        "variable": "variable_name",
        "parameter": "parameter_name",
        "property": "property_name",
        "function": "function_name",
        "class": "class_name",
    }

    _DEFAULT_TARGET_KEYWORDS = ["variable", "parameter"]

    _SCOPE_LABELS = {
        "root",
        "function_scope",
        "class_scope",
        "block_scope",
        "loop_scope",
    }

    _DECLARATION_FIELDS = {
        "left",
        "name",
        "declarator",
        "parameter",
        "pattern",
        "function",
    }

    def _is_scope_node(self, node: Node) -> bool:
        """Return whether the node introduces a new identifier scope."""
        return node.semantic_label in self._SCOPE_LABELS

    def _resolve_target_labels(self, targets: list[str]) -> set[str]:
        """Resolve user-facing target keywords into semantic labels.

        Args:
            targets: Optional target keyword list such as ``variable`` or
                ``parameter``.

        Raises:
            ValueError: If one or more target keywords are unsupported.

        Returns:
            A set of semantic labels allowed for renaming.
        """
        keywords = targets or self._DEFAULT_TARGET_KEYWORDS
        allowed_labels: set[str] = set()

        invalid_keywords: list[str] = []
        for keyword in keywords:
            label = self._TARGET_TO_LABEL.get(keyword)
            if label is None:
                invalid_keywords.append(keyword)
                continue
            allowed_labels.add(label)

        if invalid_keywords:
            valid_keywords = ", ".join(sorted(self._TARGET_TO_LABEL.keys()))
            invalid = ", ".join(sorted(set(invalid_keywords)))
            raise ValueError(
                f"Unsupported rename target keyword(s): {invalid}. "
                f"Supported keywords: {valid_keywords}."
            )

        return allowed_labels

    def _is_declaration_identifier(self, node: Node) -> bool:
        """Return whether the identifier creates or updates scope bindings."""
        return node.field in self._DECLARATION_FIELDS

    def _is_renamable_identifier(self, node: Node) -> bool:
        """Return whether the identifier is eligible for renaming."""
        return (
            node.type == "identifier"
            and bool(node.text)
            and node.semantic_label in self.allowed_labels
        )

    def _lookup_visible_name(self, original_name: str) -> str | None:
        """Look up a visible renamed binding from inner to outer scopes.

        Args:
            original_name: The original identifier text.

        Returns:
            The mapped renamed text if visible in current scope stack,
            otherwise None.
        """
        for scope in reversed(self.scope):
            renamed = scope.get(original_name)
            if renamed is not None:
                return renamed
        return None

    def _make_scoped_name(self, node: Node, original_name: str) -> str:
        """Generate a deterministic scoped rename.

        Adds a scope-depth suffix when shadowing an outer name to keep
        inner-scope declarations distinct.
        """
        rename_by_level = self._RENAME_LEVEL.get(self.level)
        base_name = (
            rename_by_level(node, self.language)
            if rename_by_level
            else _build_appendage_name(node, self.language)
        )

        if not self.scope:
            return base_name

        current_scope = self.scope[-1]
        if original_name in current_scope:
            return current_scope[original_name]

        visible_outer_name = self._lookup_visible_name(original_name)
        if visible_outer_name is not None:
            depth = len(self.scope) - 1
            return f"{base_name}_s{depth}"

        return base_name

    def apply(self, root: Node) -> list[MutationRecord]:
        """Apply identifier renaming to an annotated CST root.

        Args:
            root: Annotated CST root node.

        Returns:
            A list of MutationRecord entries for performed rename actions.
        """
        if root is None:
            return []

        self.language = ROOT_TO_LANGUAGE.get(root.type, "")

        records: list[MutationRecord] = []

        self.scope = []
        stack: list[tuple[Node, bool]] = [(root, False)]

        while stack:
            node, is_exit = stack.pop()

            if is_exit:
                if self._is_scope_node(node) and self.scope:
                    self.scope.pop()
                continue

            if self._is_scope_node(node):
                self.scope.append({})
                stack.append((node, True))

            if self._is_renamable_identifier(node) and node.text:
                original_name = node.text
                current_scope = self.scope[-1] if self.scope else None

                if self._is_declaration_identifier(node):
                    if current_scope is not None and original_name not in current_scope:
                        current_scope[original_name] = self._make_scoped_name(node, original_name)
                    new_name = (
                        current_scope[original_name]
                        if current_scope is not None
                        else self._make_scoped_name(node, original_name)
                    )
                else:
                    existing_name = self._lookup_visible_name(original_name)
                    if existing_name is not None:
                        new_name = existing_name
                    else:
                        fallback_name = self._make_scoped_name(node, original_name)
                        if current_scope is not None:
                            current_scope[original_name] = fallback_name
                        new_name = fallback_name

                node.text = new_name
                metadata = {"new_val": new_name}
                record = MutationRecord(node.start_point, MutationAction.RENAME, metadata)
                records.append(record)

            for child in reversed(node.children):
                stack.append((child, False))

        self.scope = []

        return records
