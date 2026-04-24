"""Shared tree traversal ScopeManager and scope stack helpers."""

from typing import Any


class ScopeManager:
    """Track identifier bindings across nested scopes.

    Maintains a stack of dictionaries, one per scope level.
    Lookups resolve from innermost to outermost scope.
    """

    def __init__(self) -> None:
        self._scopes: list[dict[str, str]] = []

    def enter_scope(self) -> None:
        """Push a new empty scope."""
        self._scopes.append({})

    def exit_scope(self) -> None:
        """Pop the innermost scope."""
        if self._scopes:
            self._scopes.pop()

    def declare(self, name: str, value: str) -> None:
        """Bind a value in the current (innermost) scope."""
        if self._scopes:
            self._scopes[-1][name] = value

    def resolve(self, key: str) -> Any:
        """Look up a value from inner to outer scope, returning None if absent."""
        for scope in reversed(self._scopes):
            if key in scope:
                return scope[key]
        return None

    def depth(self) -> int:
        """Return the current nesting depth."""
        return len(self._scopes)

    def current(self) -> dict[str, str] | None:
        """Return the innermost scope dict, or None if no scopes exist."""
        return self._scopes[-1] if self._scopes else None

    def reset(self) -> None:
        """Clear all scopes."""
        self._scopes = []

    def __bool__(self) -> bool:
        return bool(self._scopes)
