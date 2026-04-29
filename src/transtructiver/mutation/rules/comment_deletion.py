"""comment_deletion.py

Defines the CommentDeletion mutation rule, which removes comment nodes
from a Concrete Syntax Tree (CST) proportionally based on a mutation level. 
Each deletion generates a MutationRecord capturing the original source 
coordinates and content of the removed comment.
"""

import random
from typing import Dict, List

from transtructiver.mutation.mutation_context import MutationContext
from .mutation_rule import MutationRule, MutationRecord
from ...node import Node


class CommentDeletionRule(MutationRule):
    """
    Concrete mutation rule that deletes comment nodes from a CST.

    The number of comments deleted is proportional to the mutation level.
    Each deletion generates a MutationRecord with the node's original coordinates.

    Attributes:
        rule_name (str): CLI identifier for the rule.
        LEVEL_RATIOS (Dict[int, float]): Mapping of levels to deletion density.
    """

    # CLI rule name (used by the auto-discovery in cli.py).
    rule_name = "comment-deletion"

    # Mapping of levels to the percentage of comments to be deleted.
    LEVEL_RATIOS: Dict[int, float] = {
        0: 0.10,  # Minimal: 10% of comments
        1: 0.35,  # Low: 35% of comments
        2: 0.65,  # Medium: 65% of comments
        3: 1.00,  # Maximum: 100% of comments
    }

    def __init__(self, level: int = 0, seed: int = 42):
        """
        Initialize the CommentDeletionRule with a specified level and random seed.

        Args:
            level (int): The mutation level (0-3) determining the percentage of comments to delete.
            seed (int): Random seed used to initialize the RNG for reproducible sampling.
        """
        super().__init__()
        self._level = level
        self._rng = random.Random(seed)

    def apply(self, root: Node, context: MutationContext) -> List[MutationRecord]:
        """
        Apply the CommentDeletion mutation rule to the CST proportionally.

        Collects all eligible comment nodes, determines the number to delete
        based on the level, and performs the deletions using a seeded RNG.

        Args:
            root (Node): The root node of the CST to mutate.
            context (MutationContext): Context for tracking mutation state.

        Returns:
            List[MutationRecord]: A list of all deletions performed, each containing
                the original coordinates and text content of the removed comment.
        """
        if root is None:
            return []

        records: List[MutationRecord] = []
        comment_targets: List[tuple[Node, Node]] = []

        for node in root.traverse():
            for child in node.children:
                if child.semantic_label in ["line_comment", "block_comment"]:
                    comment_targets.append((node, child))

        if not comment_targets:
            return []

        ratio = self.LEVEL_RATIOS.get(self._level, 0.10)
        num_to_delete = max(1, round(len(comment_targets) * ratio))
        selected_deletions = self._rng.sample(comment_targets, num_to_delete)

        for parent, child in selected_deletions:
            records.append(self.record_delete(parent, child))

        return records
