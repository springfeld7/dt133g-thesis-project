"""Replacement generation facade for the comment normalization mutation rule."""

from typing import Callable

from ....node import Node
from ._context_mapping import _replace_context_mapping
from ._format_only import _replace_format_only

_ReplacementStrategy = Callable[["Node", "Node"], str]


# Strategy table keyed by replacement "level"; level 0 keeps the comment's own content.
_LEVEL_STRATEGIES: dict[int, _ReplacementStrategy] = {
    0: _replace_format_only,
    1: _replace_context_mapping,
}


class ReplacementGenerator:
    """Generate replaced comment text using a configurable replacement strategy.

    The strategy is selected by *level* at construction time, defaulting to
    the format-only normalization strategy (level 0).  New levels can be
    registered in ``_LEVEL_STRATEGIES`` without touching this class.
    """

    def __init__(self, level: int = 0) -> None:
        self._strategy: _ReplacementStrategy = _LEVEL_STRATEGIES.get(level, _replace_format_only)

    def get_replacement(self, node: Node, ancestor: Node) -> str:
        """Return the replaced text for *node*."""
        return self._strategy(node, ancestor)
