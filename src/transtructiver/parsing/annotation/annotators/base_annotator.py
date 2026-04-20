"""Shared base annotator for language-specific semantic labeling."""

from abc import ABC
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import ClassVar

from ....node import Node
from ....utils.scope_manager import ScopeManager
from ..builtin_checker import is_builtin
from .. import annotation_utils as au


# Canonical suffix map that collapses equivalent type names to shared tokens.
_TYPES = {
    "set": "set",
    **dict.fromkeys(["tuple", "pair", "triplet"], "tuple"),
    **dict.fromkeys(["list", "array"], "list"),
    **dict.fromkeys(["dict", "dictionary", "map"], "map"),
    **dict.fromkeys(["str", "string"], "string"),
    **dict.fromkeys(["int", "integer", "float", "double", "number"], "number"),
    **dict.fromkeys(["bool", "boolean"], "boolean"),
}


_DEBUG = False


_SCOPE_LABELS = {
    "block_scope",
    "class_scope",
    "condition_scope",
    "function_scope",
    "loop_scope",
    "namespace_scope",
}


_DECLARATION_FIELDS = {
    "name",
    "declarator",
    "parameter",
    "pattern",
    "left",
    "attribute",
}


@dataclass
class _AnnotateFrame:
    node: Node
    child_index: int = 0
    entered_scope: bool = False
    introduced_import_scope: bool = False
    declaration_context: bool = False
    declaration_type: str | None = None
    declaration_nodes: list[Node] = field(default_factory=list)


class BaseAnnotator(ABC):
    """Common annotation flow with map-driven language hooks."""

    _REGISTRY: ClassVar[dict[str, type["BaseAnnotator"]]] = {}
    _LANGUAGE_ALIASES: ClassVar[dict[str, str]] = {
        "c++": "cpp",
    }

    language: str
    direct_type_labels: Mapping[str, str] = {}
    direct_field_labels: Mapping[str, str] = {
        "body": "block_scope",
        "parameters": "parameter_scope",
    }
    parent_type_labels: Mapping[str, str] = {}
    identifier_types: frozenset[str] = frozenset({"identifier"})
    naming_boundary_types: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.ScopeManager = ScopeManager()
        language = getattr(cls, "language", None)
        if language:
            BaseAnnotator._REGISTRY[cls.normalize_language_key(language)] = cls

    @classmethod
    def normalize_language_key(cls, language: str) -> str:
        normalized = language.strip().lower()
        return cls._LANGUAGE_ALIASES.get(normalized, normalized)

    @classmethod
    def for_language(cls, language: str) -> "BaseAnnotator":
        annotator_cls = cls._REGISTRY.get(cls.normalize_language_key(language))
        if annotator_cls is None:
            known = sorted(cls._REGISTRY.keys())
            raise ValueError(
                f"No annotator registered for language '{language}'. Available annotators: {known}"
            )

        return annotator_cls()

    def annotate(self, root: Node, profile: dict[str, str]) -> Node:
        """Annotate a tree using a single stack for traversal and declaration context."""
        if not root:
            return root

        self._debug(f"START annotate root={root.type} text={root.text!r}")
        self.ScopeManager.reset()

        import_scope_depth = 0
        self._in_import_scope = False
        stack: list[_AnnotateFrame] = [_AnnotateFrame(node=root)]

        while stack:
            frame = stack[-1]
            node = frame.node

            if frame.child_index == 0:
                self._debug(
                    f"ENTER node={node.type} text={node.text!r} field={node.field!r} parent={node.parent.type if node.parent else None}"
                )
                scope_label = self.direct_type_labels.get(node.type) or (
                    self.direct_field_labels.get(node.field) if node.field else None
                )
                entered_scope = (not node.parent) or (scope_label in _SCOPE_LABELS)
                if entered_scope:
                    self._debug(f"  scope_enter node={node.type}")
                    self.ScopeManager.enter_scope()
                frame.entered_scope = entered_scope

                frame.declaration_context = self._is_declaration_context(node)
                self._debug(f"  declaration_context={frame.declaration_context}")
                self._in_import_scope = import_scope_depth > 0

                # Pre-order work: annotate current node before visiting children.
                self._annotate_node(node, profile)
                label = node.semantic_label
                self._debug(f"  semantic_label={label!r}")
                if label:
                    if label == "import_scope":
                        import_scope_depth += 1
                        self._in_import_scope = True
                        frame.introduced_import_scope = True

                inferred_type = self._infer_type_token(node)
                self._debug(f"  inferred_type={inferred_type!r}")
                # Find nearest declaration context frame in stack (excluding current frame)
                nearest_decl_frame = None
                for prev_frame in reversed(stack[:-1]):
                    if prev_frame.declaration_context:
                        nearest_decl_frame = prev_frame
                        break

                if inferred_type and nearest_decl_frame:
                    if nearest_decl_frame.declaration_type is None:
                        nearest_decl_frame.declaration_type = inferred_type
                        self._debug(
                            f"  bind declaration_type={inferred_type!r} to frame node={nearest_decl_frame.node.type} text={nearest_decl_frame.node.text!r}"
                        )
                    else:
                        self._debug(
                            f"  keep existing declaration_type={nearest_decl_frame.declaration_type!r}; ignore inferred_type={inferred_type!r}"
                        )

                if au.is_identifier_candidate(node):
                    if self._is_declaration_identifier(node):
                        self._debug("  identifier is declaration identifier")
                        if nearest_decl_frame:
                            nearest_decl_frame.declaration_nodes.append(node)

                        inherited_type = self._nearest_declaration_type(stack)
                        self._debug(f"  inherited_type={inherited_type!r}")
                        if inherited_type:
                            node.context_type = inherited_type
                            if node.text:
                                self.ScopeManager.declare(node.text, inherited_type)
                            self._debug(
                                f"  assign context_type={inherited_type!r} to node={node.text!r}"
                            )
                        else:
                            sibling_type = self._infer_assignment_sibling_type(node)
                            if sibling_type:
                                node.context_type = sibling_type
                                if node.text:
                                    self.ScopeManager.declare(node.text, sibling_type)
                                self._debug(
                                    f"  assign context_type={sibling_type!r} from sibling to node={node.text!r}"
                                )

                    elif node.text:
                        current_scope = self.ScopeManager.current() or {}
                        if node.text in current_scope:
                            resolved_type = current_scope[node.text]
                        else:
                            resolved_type = self.ScopeManager.resolve(node.text)
                        self._debug(
                            f"  identifier lookup text={node.text!r} resolved_type={resolved_type!r}"
                        )
                        if resolved_type:
                            node.context_type = resolved_type
                            self._debug(
                                f"  assign context_type={resolved_type!r} to node={node.text!r}"
                            )

            if frame.child_index < len(au.named_children(node)):
                child = au.named_children(node)[frame.child_index]
                frame.child_index += 1
                self._debug(
                    f"  DESCEND from={node.type} to={child.type} index={frame.child_index - 1}"
                )
                stack.append(_AnnotateFrame(node=child))
                continue

            stack.pop()
            self._debug(f"EXIT node={node.type} text={node.text!r}")

            if frame.introduced_import_scope and import_scope_depth:
                import_scope_depth -= 1

            if frame.entered_scope:
                # Clear declaration_type for the nearest declaration context frame if any
                for prev_frame in reversed(stack):
                    if prev_frame.declaration_context:
                        prev_frame.declaration_type = None
                        break

                self.ScopeManager.exit_scope()
                self._debug(f"  scope_exit node={node.type}")

            self._in_import_scope = import_scope_depth > 0

        return root

    def _annotate_node(self, node: Node, profile: dict[str, str]) -> None:
        self._debug(f"_annotate_node type={node.type} text={node.text!r} field={node.field!r}")
        if node.type in {"whitespace", "newline"} or not node.is_named:
            self._debug("  skip non-semantic node")
            return

        direct_label = self.direct_type_labels.get(node.type) or (
            self.direct_field_labels.get(node.field) if node.field else None
        )
        if direct_label:
            node.semantic_label = direct_label
            self._debug(f"  direct_label={direct_label!r}")
            return

        parent = node.parent
        if not parent:
            node.semantic_label = "root"
            self._debug("  root node labeled root")
            return

        if self.handle_special_node(node, parent):
            self._debug("  handled by special node handler")
            return

        if au.is_identifier_candidate(node):
            if self._in_import_scope:
                node.semantic_label = "import_name"
                # self._debug("  identifier inside import scope -> import_name")
                return

            self._annotate_identifier(node, parent)

        if au.is_type_like_node(node):
            if not node.semantic_label:
                node.semantic_label = "type_name"

        if node.text and is_builtin(node.text, profile):
            node.builtin = True
            self._debug(f"  builtin=True text={node.text!r}")

    def _annotate_identifier(self, node: Node, parent: Node) -> None:
        self._debug(
            f"_annotate_identifier node={node.text!r} type={node.type} parent={parent.type} field={node.field!r}"
        )
        parent_label = self.parent_type_labels.get(parent.type)
        if parent_label:
            node.semantic_label = parent_label
            self._debug(f"  parent_label={parent_label!r}")
            return

        special_label = self.handle_special_identifier(node, parent)
        if special_label:
            node.semantic_label = special_label
            self._debug(f"  special_label={special_label!r}")
            return

        node.semantic_label = self.fallback_identifier_label(node)
        self._debug(f"  fallback_label={node.semantic_label!r}")

    def handle_special_node(self, node: Node, parent: Node) -> bool:
        """Handle rare syntax-specific node cases."""
        return False

    def handle_special_identifier(self, node: Node, parent: Node) -> str | None:
        """Handle rare identifier-specific cases before the shared maps."""
        return None

    def fallback_identifier_label(self, node: Node) -> str | None:
        """Default identifier fallback label."""
        return (
            "type_name"
            if (node.field and "type" in node.field) or "type" in node.type
            else "variable_name"
        )

    def _is_declaration_context(self, node: Node) -> bool:
        parent = node.parent

        if not parent:
            return False

        if au.is_scope_node(parent, self.direct_type_labels):
            parent_label = self.direct_type_labels.get(parent.type) or (
                self.direct_field_labels.get(parent.field) if parent.field else None
            )
            if parent_label and any(t in parent_label for t in ["return", "operation", "call"]):
                return False
            return True
        return False

    def _infer_assignment_sibling_type(self, node: Node) -> str | None:
        """Infer declaration type from assignment siblings, preferring right-hand side."""
        if not node.parent:
            return None

        siblings = [s for s in au.named_children(node.parent) if s is not node]
        right_sibling = next((s for s in siblings if s.field == "right"), None)
        candidates = [right_sibling] if right_sibling else siblings

        for sibling in candidates:
            if sibling is None:
                continue

            inferred_type = self._infer_type_token(sibling)
            if inferred_type:
                return inferred_type

            inferred_type = self._match_type(sibling.type.lower(), "")
            if inferred_type:
                return inferred_type

        return None

    def _is_declaration_identifier(self, node: Node) -> bool:
        parent = node.parent
        parent_label = parent.semantic_label if parent else None

        if parent_label and "operation" in parent_label:
            return False

        if node.field in _DECLARATION_FIELDS:
            return True

        if parent and parent.type in self.parent_type_labels:
            return True

        return False

    def _nearest_declaration_type(self, stack: list[_AnnotateFrame]) -> str | None:
        self._debug(f"_nearest_declaration_type frames={len(stack)}")
        for frame in reversed(stack):
            if frame.declaration_context and frame.declaration_type:
                self._debug(
                    f"  found declaration_type={frame.declaration_type!r} on node={frame.node.type}"
                )
                return frame.declaration_type

        self._debug("  no declaration type found")
        return None

    def _infer_type_token(self, node: Node) -> str | None:
        self._debug(f"_infer_type_token node={node.type} text={node.text!r} field={node.field!r}")
        if not au.is_type_like_node(node):
            self._debug("  not type-like")
            return None

        raw = self._resolve_type_text(node)
        if not raw:
            raw = node.text or node.type

        raw = raw.strip().lower()
        if not raw:
            self._debug("  no raw type text")
            return None

        terminal = self._extract_terminal_type(raw, node.parent.type if node.parent else None)
        matched = self._match_type(raw, terminal)
        self._debug(f"  raw={raw!r} terminal={terminal!r} matched={matched!r}")
        return matched

    def _resolve_type_text(self, type_node: Node) -> str | None:
        self._debug(f"_resolve_type_text node={type_node.type} text={type_node.text!r}")
        if type_node.text:
            self._debug(f"  direct_text={type_node.text!r}")
            return type_node.text

        for child in au.named_children(type_node):
            if child.text and (
                child.type == "identifier"
                or "type" in child.type
                or (child.field and "type" in child.field)
                or not child.children
            ):
                self._debug(f"  child_text={child.text!r} child_type={child.type}")
                return child.text

        for nested in type_node.traverse():
            if nested.text and (
                nested.type == "identifier"
                or "type" in nested.type
                or (nested.field and "type" in nested.field)
                or not nested.children
            ):
                self._debug(f"  nested_text={nested.text!r} nested_type={nested.type}")
                return nested.text

        self._debug("  no type text found")
        return None

    def _debug(self, message: str) -> None:
        if _DEBUG:
            print(f"[annotate] {message}")
        return

    def _match_type(self, lowered: str, terminal: str) -> str | None:
        exact = _TYPES.get(terminal)
        if exact:
            return exact

        for key in sorted(_TYPES.keys(), key=len, reverse=True):
            if key in lowered:
                return _TYPES[key]

        return None

    def _extract_terminal_type(self, text: str, parent_type: str | None) -> str:
        normalized = text.strip()
        if not normalized:
            return ""

        if parent_type == "method_reference":
            normalized = normalized.split("::", 1)[0]

        if "<" in normalized:
            normalized = normalized.split("<", 1)[0]

        normalized = normalized.rsplit("::", 1)[-1].split(".")[-1]
        return normalized.strip()
