"""mutation_engine.py

The MutationEngine manages the transformation lifecycle by sequentially applying rules 
and recording every change in a centralized MutationManifest. This provides 
a verifiable history of how each node in the tree was modified.
"""

from collections import defaultdict
import heapq
from typing import List
from .rules.mutation_rule import MutationRecord, MutationRule
from .mutation_manifest import MutationManifest
from .mutation_context import MutationContext
from ..node import Node


class MutationEngine:
    """
    Engine for applying mutation rules to transform syntax trees.

    The MutationEngine takes a list of mutation rules and applies them sequentially
    to a CST. It aggregates local changes into a global MutationManifest keyed
    by the original source coordinates of the nodes.

    Attributes:
        rules (List[MutationRule]): Sequence of rules to be executed.
    """

    # Define rule dependencies to ensure correct execution order
    _RULE_DEPENDENCIES = {
        "dead-code-insertion": ["whitespace-normalization"],
        "control-structure-substitution": ["whitespace-normalization"],
        "rename-identifier": ["dead-code-insertion", "control-structure-substitution"],
    }

    def __init__(self, rules: List[MutationRule]):
        """
        Initialize the MutationEngine with a list of rules.

        Args:
            rules (List[MutationRule]): List of mutation rule objects.
        """
        self.rules: List[MutationRule] = rules
        self.manifest = MutationManifest()
        self.context = MutationContext()

    def apply_mutations(self, cst: Node) -> MutationManifest:
        """
        Apply all mutation rules to the given CST.

        Rules are applied sequentially. This method populates the transformation Manifest
        by aggregating mutation records from each rule, which serves as the
        source of truth for downstream verification. The result is also stored
        as ``self.manifest`` for post-hoc inspection by the caller.

        Args:
            cst (Node): The root node of the CST to mutate.

        Returns:
            MutationManifest: The complete transformation Manifest.
        """
        manifest = MutationManifest()
        self.context.reset()

        ordered_rules = self._order_rules()

        for rule in ordered_rules:
            local_changes = rule.apply(cst, self.context)
            self._merge_to_manifest(manifest, local_changes, rule.name)

        self.manifest = manifest
        return manifest

    def _order_rules(self) -> List[MutationRule]:
        """Order rules using dependency constraints (topological sort).

        Only enforces constraints between rules that are present.
        Preserves original order where no constraints apply.

        Returns:
            List[MutationRule]: Rules ordered according to dependencies.
        """

        # Preserve original order as tie-breaker
        original_order = {r.name: i for i, r in enumerate(self.rules)}

        # Map rule_name -> rule instance
        rule_map = {r.name: r for r in self.rules}

        in_degree = {name: 0 for name in rule_map}
        graph = defaultdict(list)

        # Build graph ONLY for rules that exist in this run
        for rule, deps in self._RULE_DEPENDENCIES.items():
            if rule not in rule_map:
                continue

            for dep in deps:
                if dep not in rule_map:
                    continue

                graph[dep].append(rule)
                in_degree[rule] += 1

        # Priority queue ensures stable ordering based on original input order
        queue = [(original_order[name], name) for name in rule_map if in_degree[name] == 0]
        heapq.heapify(queue)

        ordered_names = []

        while queue:
            _, current = heapq.heappop(queue)
            ordered_names.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(queue, (original_order[neighbor], neighbor))

        # Detect cycles (should never happen unless config is wrong)
        if len(ordered_names) != len(rule_map):
            raise ValueError("Cycle detected in rule dependencies")

        return [rule_map[name] for name in ordered_names]

    def _merge_to_manifest(
        self, manifest: MutationManifest, changes: List[MutationRecord], rule_name: str
    ) -> None:
        """
        Converts MutationRecords into ManifestEntries and merges them into
        a mutation manifest.

        Args:
            manifest: The mutation manifest to merge changes into.
            changes: A list of mutations produced by a single rule execution.
            rule_name: The name of the rule, used for audit history.
        """
        for record in changes:
            manifest.add_entry(
                node_id=record.node_id,
                action=record.action,
                metadata=record.metadata,
                rule_name=rule_name,
            )
