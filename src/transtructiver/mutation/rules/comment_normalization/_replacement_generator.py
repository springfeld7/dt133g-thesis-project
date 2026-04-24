"""Replacement generation facade for the comment normalization mutation rule."""

from typing import Callable

from ....node import Node
from ._context_mapping import _replace_context_mapping

_ReplacementStrategy = Callable[["Node", "Node"], str]

# Strategy table keyed by replacement "level"; level 0 is the current default heuristic.
_LEVEL_STRATEGIES: dict[int, _ReplacementStrategy] = {
    0: _replace_context_mapping,
}


class ReplacementGenerator:
    """Generate replaced comment text using a configurable replacement strategy.

    The strategy is selected by *level* at construction time, defaulting to
    the context mapping heuristic (level 0).  New levels can be registered in
    ``_LEVEL_STRATEGIES`` without touching this class.
    """

    def __init__(self, level: int = 0) -> None:
        self._strategy: _ReplacementStrategy = _LEVEL_STRATEGIES.get(
            level, _replace_context_mapping
        )

    def get_replacement(self, node: Node, ancestor: Node) -> str:
        """Return the replaced text for *node*."""
        return self._strategy(node, ancestor)
