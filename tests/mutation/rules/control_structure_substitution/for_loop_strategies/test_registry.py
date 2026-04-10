"""Unit tests for the For-Loop Strategy Registry."""

import pytest

from transtructiver.mutation.rules.control_structure_substitution.for_loop_strategies.registry import (
    get_for_loop_strategy,
    _STRATEGY_REGISTRY,
)
from transtructiver.mutation.rules.control_structure_substitution.for_loop_strategies.cstyle_strategy import (
    CStyleForLoopStrategy,
)
from transtructiver.mutation.rules.control_structure_substitution.for_loop_strategies.python_strategy import (
    PythonForLoopStrategy,
)


class TestForLoopStrategyRegistry:
    """Test suite for validating the for-loop strategy registry behavior."""

    def test_get_for_loop_strategy_returns_correct_types(self):
        """Verify the registry returns the expected strategy instances."""
        assert isinstance(get_for_loop_strategy("python"), PythonForLoopStrategy)
        assert isinstance(get_for_loop_strategy("java"), CStyleForLoopStrategy)
        assert isinstance(get_for_loop_strategy("cpp"), CStyleForLoopStrategy)

    def test_get_for_loop_strategy_is_case_insensitive(self):
        """Verify lookup handles mixed and upper casing correctly."""
        assert isinstance(get_for_loop_strategy("pYtHoN"), PythonForLoopStrategy)
        assert isinstance(get_for_loop_strategy("JAVA"), CStyleForLoopStrategy)

    def test_get_for_loop_strategy_strips_whitespace(self):
        """Verify lookup strips surrounding whitespace before resolving."""
        assert isinstance(get_for_loop_strategy("  cpp  "), CStyleForLoopStrategy)

    def test_registry_contains_all_mapped_languages(self):
        """Ensure all registered languages can be resolved without failure."""
        for lang in _STRATEGY_REGISTRY.keys():
            strategy = get_for_loop_strategy(lang)
            assert strategy is not None

    def test_get_for_loop_strategy_raises_for_unknown_language(self):
        """
        Verify unknown languages raise ValueError instead of returning None.
        """
        with pytest.raises(ValueError) as excinfo:
            get_for_loop_strategy("nolanguage")

        assert "nolanguage" in str(excinfo.value)

    def test_get_for_loop_strategy_raises_for_empty_string(self):
        """Verify empty input triggers a ValueError."""
        with pytest.raises(ValueError):
            get_for_loop_strategy("")

    def test_get_for_loop_strategy_raises_for_none(self):
        """Verify None input triggers a ValueError."""
        with pytest.raises(ValueError):
            get_for_loop_strategy(None)  # type: ignore
