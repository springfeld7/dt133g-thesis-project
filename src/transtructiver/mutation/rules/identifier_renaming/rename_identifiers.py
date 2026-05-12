"""rename_identifiers.py

Identifier renaming mutation rule.

This module defines RenameIdentifiersRule, a concrete MutationRule that applies
deterministic identifier renaming over annotated CST nodes and emits
MutationRecord entries for each rename action.
"""

import random
import string

from ....node import Node
from ...mutation_context import MutationContext
from ..mutation_rule import MutationRecord, MutationRule
from ....utils.scope_manager import ScopeManager
from ._name_generator import NameGenerator


class RenameIdentifiersRule(MutationRule):
    """Rename eligible identifier nodes using a generated naming scheme.

    The rule targets semantic identifier categories (variables, parameters,
    properties, functions, classes), applies language-aware suffix formatting,
    and preserves scope consistency for declarations and references.

    Orchestrates three focused components:
    - ``ScopeManager``: scope enter/exit and name binding lookup
    - ``NameGenerator``: base name and suffix generation
    - ``apply()``: CST traversal and MutationRecord emission
    """

    # CLI rule name (used by the auto-discovery in cli.py).
    rule_name = "rename-identifier"

    # Public target keywords mapped to the semantic labels produced by annotation.
    _TARGET_TO_LABEL = {
        "variable": "variable_name",
        "parameter": "parameter_name",
        "property": "property_name",
        "function": "function_name",
        "class": "class_name",
    }

    # Default rename categories when the caller does not specify targets.
    _DEFAULT_TARGET_KEYWORDS = ["variable", "parameter"]

    # Semantic labels that introduce a fresh naming scope during traversal.
    _SCOPE_LABELS = {
        "root",
        "function_scope",
        "class_scope",
        "condition_scope",
        "loop_scope",
    }

    # Node field names treated as declarations rather than references.
    _DECLARATION_FIELDS = {
        "name",
        "declarator",
        "parameter",
        "pattern",
        "function",
        "left",
        "attribute",
    }

    def __init__(self, level: int = 0, targets: list[str] | None = None) -> None:
        """Initialize the rule configuration and helper components.

        Args:
            level: Naming strategy level passed to :class:`NameGenerator`.
            targets: Optional rename target keywords such as ``variable``,
                ``parameter``, ``property``, ``function``, or ``class``.
        """
        super().__init__()
        self.level = level
        self.targets = targets or []
        # Precompute rename filters and helper objects used during apply().
        self.allowed_labels = self._resolve_target_labels(self.targets)
        self._scope = ScopeManager()
        self._namer = NameGenerator(level)

    # ------------------------------------------------------------------
    # Backward-compatible public view of the scope stack
    # ------------------------------------------------------------------

    @property
    def scope(self) -> list[dict[str, str]]:
        """Current scope stack (empty between apply() calls)."""
        return self._scope._scopes

    # ------------------------------------------------------------------
    # Predicate helpers
    # ------------------------------------------------------------------

    def _is_scope_node(self, node: Node) -> bool:
        """Return whether the node introduces a new identifier scope."""
        return node.semantic_label in self._SCOPE_LABELS

    def _is_declaration_identifier(self, node: Node) -> bool:
        """Return whether the identifier creates or updates scope bindings."""
        return node.field in self._DECLARATION_FIELDS

    def _is_renamable_identifier(self, node: Node) -> bool:
        """Return whether the identifier is eligible for renaming."""
        return (
            "identifier" in node.type
            and not bool(node.builtin)
            and bool(node.text)
            and node.semantic_label in self.allowed_labels
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def _make_scoped_name(self, node: Node, _original_name: str, language: str) -> str:
        """Generate a deterministic rename for the current identifier."""
        return self._namer.make_name(node, language)

    def _name_exists_in_visible_scope(self, candidate: str) -> bool:
        """Return whether a generated name already exists in any active scope."""
        return any(candidate in scope.values() for scope in self._scope._scopes)

    def _bump_name(self, base: str, index: int, language: str) -> str:
        """Create a deterministic disambiguated variant of ``base``."""
        if language == "python":
            return f"{base}_{index}"
        return f"{base}{index}"

    def _resolve_name(self, node: Node, original_name: str, language: str) -> str:
        """Return the new name for an identifier, registering it in scope as needed."""
        if self._is_declaration_identifier(node):
            current = self._scope.current()
            if current is not None and original_name in current:
                return current[original_name]

            outer_existing = self._scope.resolve(original_name)

            # Targeted shadow handling: when a parameter declaration shadows
            # an outer binding (e.g., class field), keep a distinct local name.
            if node.semantic_label == "parameter_name" and outer_existing is not None:
                base_name = self._make_scoped_name(node, original_name, language)
                new_name = base_name
                bump_index = 1
                while new_name == outer_existing or self._name_exists_in_visible_scope(new_name):
                    if self.level == 1:
                        new_name = self._make_scoped_name(node, original_name, language)
                    if self.level == 3:
                        new_name = random.choice(string.ascii_lowercase)
                    new_name = self._bump_name(base_name, bump_index, language)
                    bump_index += 1
                self._scope.declare(original_name, new_name)
                return new_name

            if outer_existing is not None:
                return outer_existing

            new_name = self._make_scoped_name(node, original_name, language)
            self._scope.declare(original_name, new_name)
            return new_name

        existing = self._scope.resolve(original_name)
        if existing is not None:
            return existing

        new_name = self._make_scoped_name(node, original_name, language)
        self._scope.declare(original_name, new_name)
        return new_name

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def apply(self, root: Node, context: MutationContext) -> list[MutationRecord]:
        """Apply identifier renaming to an annotated CST root.

        Args:
            root: Annotated CST root node.
            context: MutationContext for this mutation application,
                     unused by this rule but available for future extensions.

        Returns:
            A list of MutationRecord entries for performed rename actions.

        Raises:
            ValueError: If root.language is missing.
        """
        if root is None:
            return []

        language = root.language.lower() if root.language else None
        if language is None:
            raise ValueError(
                "No language found on root node. "
                "Set root.language before applying RenameIdentifiersRule."
            )

        self._scope.reset()
        records: list[MutationRecord] = []
        stack: list[tuple[Node, bool]] = [(root, False)]

        while stack:
            node, is_exit = stack.pop()

            if is_exit:
                # Exit markers let a single iterative walk mirror recursive scope unwinding.
                self._scope.exit_scope()
                continue

            if self._is_scope_node(node):
                self._scope.enter_scope()
                stack.append((node, True))

            # Skip renaming for builtins
            if node.semantic_label == "builtin_name":
                for child in reversed(node.children):
                    stack.append((child, False))
                continue

            # If a binding is already known for this identifier in the current
            # scope stack, rename it regardless of semantic label coverage.
            # This keeps declaration/reference renames consistent even when
            # a reference node was not annotated as variable_name/parameter_name.
            if (
                "identifier" in node.type
                and node.text
                and not self._is_declaration_identifier(node)
            ):
                existing = self._scope.resolve(node.text)
                if existing is not None and existing != node.text:
                    original_name = node.text
                    records.append(self.record_rename(node, original_name, existing))
                    for child in reversed(node.children):
                        stack.append((child, False))
                    continue

            if not self._is_renamable_identifier(node) or not node.text:
                for child in reversed(node.children):
                    stack.append((child, False))
                continue

            # Rename after scope setup so declarations and references resolve consistently.
            original_name = node.text
            new_name = self._resolve_name(node, original_name, language)
            records.append(self.record_rename(node, original_name, new_name))

            for child in reversed(node.children):
                stack.append((child, False))

        self._scope.reset()
        return records
