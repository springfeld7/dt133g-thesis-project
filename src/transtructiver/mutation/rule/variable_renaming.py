"""variable_renaming.py

Defines VariableRenaming, a concrete MutationRule that renames identifier
nodes in a deterministic way and reports all changes as MutationRecord items.
"""

from typing import Dict, List

from ...node import Node
from ..mutation_rule import MutationRecord, MutationRule
from ..mutation_types import MutationAction, _ACTION_REQUIRED_KEYS


class VariableRenaming(MutationRule):
    """Rename identifier nodes using a generated naming scheme."""

    def __init__(self) -> None:
        super().__init__()
        self.idx = 0

    def rename(self, original_name: str) -> str:

        new_name = original_name + str(self.idx)
        self.idx += 1
        return new_name

    def apply(self, root: Node) -> List[MutationRecord]:
        """Apply identifier renaming across the provided tree root."""
        if root is None:
            return []

        records: List[MutationRecord] = []

        rename_map: Dict[str, str] = {}
        for node in root.traverse():
            if node.type != "identifier" or not node.text:
                continue

            original_name = node.text
            if original_name not in rename_map:
                rename_map[original_name] = self.rename(original_name)

            new_name = rename_map[original_name]
            node.text = new_name

            metadata = {"new_val": new_name}
            record = MutationRecord(node.start_point, MutationAction.RENAME, metadata)
            records.append(record)

        return records
