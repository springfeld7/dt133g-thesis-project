"""mutation_types.py

Centralized type definitions and constants for the Mutation Engine.
This module defines the shared 'language' used by MutationRules to report 
changes and by the SIVerifier to validate them.
"""

from enum import Enum, auto


class MutationAction(Enum):
    """
    Enumeration of supported transformation action types.

    Members:
        RENAME: Modifying the identifier name of a node.
        DELETE: Removing a node and its subtree.
        MOVE: Relocating a node to a different parent or position.
        TYPE_CHANGE: Altering the node's structural type (e.g., Loop Substitution: FOR -> WHILE).
        INSERT: Adding new synthetic nodes into the tree.
    """

    RENAME = auto()
    DELETE = auto()
    MOVE = auto()
    TYPE_CHANGE = auto()
    INSERT = auto()

    def __str__(self) -> str:
        """Returns the name of the action (e.g., 'RENAME') as a plain string."""
        return self.name
