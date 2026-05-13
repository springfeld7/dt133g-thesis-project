"""Mutation Context Module

Provides the MutationContext class to manage shared state, such as
synthetic coordinate generation and reserved identifier tracking,
across multiple mutation rules.
"""

from typing import Any
from dataclasses import dataclass, field

from evaluation.varclr.models.encoders import Encoder


@dataclass
class MutationContext:
    """Shared state for a single mutation lifecycle."""

    synthetic_row_counter: int = -1
    taken_names: set[str] = field(default_factory=set)

    tokenizer: Any | None = None
    mlm_model: Any | None = None
    varclr: Encoder | None = None

    def next_id(self) -> int:
        """
        Returns the next unique synthetic ID and decrements the counter.
        Used to ensure no two rules inject nodes with colliding coordinates.
        """
        current = self.synthetic_row_counter
        self.synthetic_row_counter -= 1
        return current

    def reset(self) -> None:
        """
        Resets fields that must not persist across all snippets.
        """
        self.synthetic_row_counter = -1
        self.taken_names.clear()
