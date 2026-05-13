"""Name generation facade for the identifier renaming mutation rule."""

from typing import Callable, Optional

from ...mutation_context import MutationContext
from ....node import Node
from ._rename_appendage import _build_appendage_name
from ._rename_substitution import _build_substitute_name
from ._rename_abbreviation import _build_abbreviated_name
from ._rename_destruction import _build_destructed_name

_NamingStrategy = Callable[[Node, str, Optional[MutationContext]], str]

# Strategy table keyed by rename "level"; level 0 is the current default heuristic.
_LEVEL_STRATEGIES: dict[int, _NamingStrategy] = {
    0: _build_appendage_name,
    1: _build_substitute_name,
    2: _build_abbreviated_name,
    3: _build_destructed_name,
}


class NameGenerator:
    """Generate renamed identifier text using a configurable naming strategy.

    The strategy is selected by *level* at construction time, defaulting to
    the appendage heuristic (level 0).  New levels can be registered in
    ``_LEVEL_STRATEGIES`` without touching this class.
    """

    def __init__(self, level: int = 0) -> None:
        self._strategy: _NamingStrategy = _LEVEL_STRATEGIES.get(level, _build_appendage_name)

    def make_name(self, node: Node, language: str, context: Optional[MutationContext]) -> str:
        """Return the renamed text for *node* in *language*."""
        return self._strategy(node, language, context)
