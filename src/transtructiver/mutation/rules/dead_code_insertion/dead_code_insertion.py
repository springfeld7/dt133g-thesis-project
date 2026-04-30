"""Rule for injecting language-specific dead code into a CST.

This module defines the InsertDeadCodeRule, which traverses a CST and identifies
safe containers (blocks, functions, etc.) for injecting unreachable code. It
leverages language-specific lexicons to ensure the injected code is syntactically
valid and uses a ScopeManager to prevent identifier collisions.
"""

from operator import indexOf
import random
from typing import List

from ....node import Node
from ..mutation_rule import MutationRule, MutationRecord
from ...mutation_context import MutationContext
from ..utils.indentation_util import IndentationUtils
from .lexicons.dead_code_lexicon import DeadCodeLexicon
from .lexicons.registry import get_lexicon
from .insertion_strategies.registry import get_strategy
from .insertion_strategies.insertion_strategy import InsertionStrategy
from ....utils.scope_manager import ScopeManager


class DeadCodeInsertionRule(MutationRule):
    """
    Injects unreachable code blocks with scope-aware identifier generation.

    The rule uses a probability-based approach governed by the 'level' attribute
    to decide where to insert code. It ensures that at least one insertion occurs
    if the tree contains valid container nodes.

    Attributes:
        rule_name (str): CLI identifier for the rule.
        _NATURAL_VAR_NAMES (List[str]): A pool of human-like variable names for injections.
        _LOOP_VAR_NAMES (List[str]): Common loop variable names for generating loop constructs.
        _SCOPE_LABELS (Set[str]): Semantic labels that denote new scopes for identifier tracking.
        _level (int): Intensity level for insertion frequency (0-3).
        _rng (random.Random): Internal RNG for deterministic behavior based on seed.
        _base_indent (str): The string used for indentation in injected code (e.g. "    " for 4 spaces).
        _scope (ScopeManager): Manages variable scopes to avoid naming collisions during injection.
    """

    rule_name = "dead-code-insertion"

    # A pool of natural variable names to choose from for injected code, enhancing plausibility.
    _NATURAL_VAR_NAMES = [
        "val",
        "temp",
        "result",
        "data",
        "item",
        "entry",
        "element",
        "content",
        "obj",
        "target",
        "source",
        "output",
        "input",
        "buffer",
        "cache",
        "count",
        "total",
        "sum",
        "min",
        "max",
        "avg",
        "offset",
        "index",
        "pos",
        "limit",
        "threshold",
        "delta",
        "factor",
        "scale",
        "ratio",
        "size",
        "length",
    ]

    # Standard loop indices
    _LOOP_VAR_NAMES = [
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]

    # Semantic labels that introduce a fresh naming scope during traversal.
    _SCOPE_LABELS = {
        "root",
        "function_scope",
        "class_scope",
        "block_scope",
        "loop_scope",
    }

    # Valid parent labels/types for block scopes where dead code can be safely injected.
    _VALID_PARENT_LABELS = {
        "loop_scope",
        "function_scope",
    }
    _VALID_PARENT_TYPES = {"if_statement", "else_clause", "for_statement", "while_statement"}

    def __init__(self, level: int = 0, seed: int = 42, indent_unit: str | None = None) -> None:
        """
        Initializes the rule with configuration for intensity and determinism.

        Args:
            level (int): Intensity level (0-3). Higher levels increase the number of insertions.
            seed (int): Seed for the internal random number generator.
            indent_unit (str): The string used for indentation in injected code,
                               if None, it will be auto-detected from the CST or default to 4 spaces.
        """
        super().__init__()
        self._level = level
        self._rng = random.Random(seed)
        self._base_indent = indent_unit
        self._scope = ScopeManager()

    def apply(self, root: Node, context: MutationContext) -> List[MutationRecord]:
        """
        Entry point for the mutation. Validates language and initializes
        traversal.

        Args:
            root (Node): The annotated CST root node.
            context (MutationContext): The mutation context for tracking state across rules.

        Returns:
            List[MutationRecord]: A list of insertion records performed.

        Raises:
            ValueError: If the root node lacks language metadata.
        """
        if root is None:
            return []

        # Determine language
        language = root.language.lower().strip() if root.language else None
        if not language:
            raise ValueError("No language found on root node.")
        # Retrieve the lexicon class for the specified language
        lexicon_cls = get_lexicon(language)
        lexicon = lexicon_cls(self._rng)

        # Retrieve the insertion strategy for the specified language
        insertion_strategy = get_strategy(language)

        # Determine indentation
        indent = self._base_indent or IndentationUtils.detect_indent_unit(root)
        lexicon.set_indent_unit(indent)

        return self._apply_traversal(root, context, insertion_strategy, lexicon)

    def _apply_traversal(
        self,
        root: Node,
        context: MutationContext,
        strategy: InsertionStrategy,
        lexicon: DeadCodeLexicon,
    ) -> List[MutationRecord]:
        """
        Iteratively walks the tree to track scopes, record identifiers,
        and perform safe code injections using language-specific strategies.

        Args:
            root (Node): The CST root node.
            context (MutationContext): Tracking state across rules.
            lexicon (DeadCodeLexicon): Generator for the code strings.
            strategy (InsertionStrategy): Structural rules for the language.

        Returns:
            List[MutationRecord]: Records of all code insertions performed.
        """
        self._scope.reset()
        candidates = self._collect_candidates(root, strategy)

        if not candidates:
            return []

        target_count = self._compute_insertion_budget(candidates)
        selected = self._select_candidates(candidates, target_count)
        return self._execute_insertion_pass(root, selected, context, strategy, lexicon)

    def _execute_insertion_pass(self, root, selected, context, strategy, lexicon):
        """
        Executes a deterministic insertion pass based on the selected candidates.

        Args:
            root (Node): The CST root node.
            selected (List[tuple[Node, str]]): The list of selected insertion points.
            context (MutationContext): The mutation context for tracking state across rules.
            strategy (InsertionStrategy): Structural rules for the language.
            lexicon (DeadCodeLexicon): Generator for the code strings.

        Returns:
            List[MutationRecord]: Records of all code insertions performed.
        """
        self._scope.reset()

        selected_targets = {id(child): (container, child) for container, child in selected}
        records = []

        # Iterative stack: (node, is_exit_marker)
        stack: list[tuple[Node, bool]] = [(root, False)]

        while stack:
            node, is_exit = stack.pop()

            # Scope exit handling
            if is_exit:
                self._scope.exit_scope()
                continue

            # Scope entry handling
            if node.semantic_label in self._SCOPE_LABELS:
                self._scope.enter_scope()
                stack.append((node, True))

            # Identifier Tracking (to avoid shadowing/clashes)
            if "identifier" in node.type and node.text:
                self._scope.declare(node.text, "exists")

            # Perform insertion if this node+child is selected
            if id(node) in selected_targets:
                container, child = selected_targets[id(node)]
                prefix = strategy.get_indent_prefix(container)

                records.append(self._inject_dead_code(child, prefix, lexicon, context))

            for child in reversed(node.children):
                stack.append((child, False))

        return records

    def _collect_candidates(
        self, root: Node, strategy: InsertionStrategy
    ) -> List[tuple[Node, Node]]:
        """
        Collects all valid insertion points for dead code injection.

        Traverses the CST to identify valid container nodes and their child nodes where dead code
        can be injected according to the provided strategy.

        Args:
            root (Node): The root of the CST to traverse.
            strategy (InsertionStrategy): Language-specific rules for determining
                valid containers, gaps, and termination points within blocks.

        Returns:
            List[tuple[Node, Node]]: A list of candidate insertion points,
            where each tuple contains:
                - container node (the block or scope containing the insertion point)
                - target node (the child node before which insertion may occur)
        """
        candidates: List[tuple[Node, Node]] = []

        for node in root.traverse():
            if not self._is_valid_container(node, strategy):
                continue

            preceding = None
            for child in node.children:
                if strategy.is_valid_gap(child, preceding):
                    candidates.append((node, child))

                if strategy.is_terminal(child):
                    break

                preceding = child

        return candidates

    def _inject_dead_code(
        self, target: Node, prefix: str, lexicon: DeadCodeLexicon, context: MutationContext
    ) -> MutationRecord:
        """
        Creates a synthetic node and records the insertion.

        Args:
            target (Node): The node before which the dead code will be inserted.
            prefix (str): The indentation prefix for the generated code.
            lexicon (DeadCodeLexicon): The lexicon to generate the dead code string.
            context (MutationContext): The mutation context for tracking state across rules.

        Returns:
            MutationRecord: The record of the insertion performed.
        """
        var_name = self._get_var_name("d")
        loop_var = self._get_loop_var()
        dead_code = lexicon.get_random_dead_code(var_name, loop_var, prefix)

        # Create an adopt dead code node
        dc_node = Node(
            start_point=(context.next_id(), -1),
            end_point=target.start_point,
            type="dead_code",
            text=dead_code,
        )
        dc_node.parent = target.parent

        # Insert into the tree
        if target.parent:
            idx = indexOf(target.parent.children, target)
            target.parent.children.insert(idx, dc_node)

        return self.record_insert(
            dc_node.start_point,
            insertion_point=target.start_point,
            new_text=dead_code,
            new_type="dead_code",
        )

    def _compute_insertion_budget(self, candidates: list) -> int:
        """
        Computes the exact number of dead code insertions based on a fixed
        insertion rate per 100 candidate insertion points, scaled by mutation level.

        The model is deterministic and uses ceiling rounding to ensure that
        small candidate sets still produce meaningful mutation when applicable.

        Insertion rates per 100 candidates by level:
            Level 0 → 2 insertions
            Level 1 → 4 insertions
            Level 2 → 8 insertions
            Level 3 → 16 insertions

        Args:
            candidates (list): List of valid insertion candidates.

        Returns:
            int: Number of insertions to perform (clamped to available candidates).
        """

        n = len(candidates)
        if n == 0:
            return 0

        per_100_rates = {
            0: 2,
            1: 4,
            2: 8,
            3: 16,
        }

        rate = per_100_rates.get(
            self._level, 2
        )  # Default to level 0 rate if level is out of bounds
        budget = round((n * rate) / 100)

        return budget if budget > 0 else 1  # Ensure at least one insertion if candidates exist

    def _select_candidates(self, candidates, k):
        """
        Randomly selects k candidates from the list of valid insertion points.

        Args:
            candidates (list): List of valid insertion candidates.
            k (int): Number of candidates to select.

        Returns:
            list: A randomly selected subset of candidates for insertion.
        """
        return self._rng.sample(candidates, k)

    def _is_valid_container(self, node: Node, strategy: InsertionStrategy) -> bool:
        """
        Checks if the node is a block that lives inside a function, loop, or conditional.

        Args:
            node (Node): The node to check for eligibility as a container for dead code insertion.
            strategy (InsertionStrategy): The language-specific insertion strategy.

        Returns:
            bool: True if the node is a valid container for insertion, False otherwise.
        """
        if node.semantic_label != "block_scope":
            return False

        if not strategy.is_valid_container(node):
            return False

        parent = node.parent
        if not parent:
            return False

        return (
            parent.semantic_label in self._VALID_PARENT_LABELS
            or parent.type in self._VALID_PARENT_TYPES
        )

    def _get_var_name(self, prefix: str) -> str:
        """
        Generates a unique name by checking the natural pool first, then falling
        back to a hex-suffixed name.

        Args:
            prefix (str): A base prefix for the variable name.

        Returns:
            str: A unique variable name that does not collide with existing identifiers in scope.
        """
        # Attempt to use a name from the natural pool first,
        # ensuring it doesn't collide with existing identifiers in scope
        for candidate in self._NATURAL_VAR_NAMES:
            if self._scope.resolve(candidate) is None:
                self._scope.declare(candidate, "injected")
                return candidate

        # Fallback if all natural names are taken in the current scope
        return self._generate_fallback_name(prefix)

    def _generate_fallback_name(self, prefix: str) -> str:
        """
        Generates a unique hex-suffixed variable name that does not collide
        with existing identifiers in the current scope.

        Args:
            prefix (str): Base string for the variable name.

        Returns:
            str: Unique variable name.
        """
        while True:
            candidate = f"{prefix}_{self._rng.getrandbits(16):x}"
            if self._scope.resolve(candidate) is None:
                self._scope.declare(candidate, "injected")
                return candidate

    def _get_loop_var(self) -> str:
        """
        Attempts to find a unique standard loop index (i, j, k, etc.).

        Returns:
            str: A human-like loop variable or a unique fallback if all are taken.
        """
        for candidate in self._LOOP_VAR_NAMES:
            if self._scope.resolve(candidate) is None:
                self._scope.declare(candidate, "injected")
                return candidate
        return self._get_var_name("idx")
