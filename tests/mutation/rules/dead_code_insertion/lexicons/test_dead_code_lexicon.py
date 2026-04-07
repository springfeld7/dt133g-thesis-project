"""Unit tests for DeadCodeLexicon base-class logic using a dummy concrete subclass."""

import random
import pytest
from src.transtructiver.mutation.rules.dead_code_insertion.lexicons.dead_code_lexicon import (
    DeadCodeLexicon,
)


# ===== Minimal Dummy Subclass =====
class DummyLexicon(DeadCodeLexicon):
    """Concrete subclass to enable instantiation of the base class."""

    OPAQUE_PREDICATES = ["false_condition"]
    UNREACHABLE_LOOP_HEADERS = ["for(int {var}=0;{var}<0;++{var})"]
    IDENTITY_OPS_STR = ["{var} = {var}"]
    IDENTITY_OPS_NUMERIC = ["{var} = {var}"]

    def get_assignment_statement(self, var_name, value):
        """Return dummy assignment for base-class testing."""
        return f"{var_name} = {value}"

    def format_block(self, header, body, prefix, is_if):
        """Return simple concatenation for base-class testing."""
        return f"{header}\n{body}"


# ===== Fixtures =====
@pytest.fixture
def rng():
    """Deterministic RNG."""
    return random.Random(42)


@pytest.fixture
def lexicon(rng):
    """Provides a DummyLexicon instance."""
    return DummyLexicon(rng)


# ===== Tests for Raw Generators =====
def test_get_raw_int_bounds(lexicon):
    """_get_raw_int returns value within range."""
    for _ in range(10):
        val = lexicon._get_raw_int()
        assert 0 <= val <= 100


def test_get_raw_float_bounds(lexicon):
    """_get_raw_float returns value within range and precision."""
    for _ in range(10):
        val = lexicon._get_raw_float()
        assert 0.1 <= val <= 99.9
        assert round(val, 2) == val


def test_get_raw_string_length(lexicon):
    """_get_raw_string returns string of correct length and chars."""
    for _ in range(5):
        s = lexicon._get_raw_string()
        assert len(s) == 5
        assert s.isalnum()


# ===== Tests for Meaningless Modification =====


def test_get_meaningless_modification_numeric(lexicon):
    """_get_meaningless_modification uses numeric template."""
    lexicon._current_type = "int"
    stmt = lexicon._get_meaningless_modification("v")
    assert stmt == "v = v"


def test_get_meaningless_modification_string(lexicon):
    """_get_meaningless_modification uses string template."""
    lexicon._current_type = "string"
    stmt = lexicon._get_meaningless_modification("v")
    assert stmt == "v = v"


# ===== Tests for Transaction Builder =====


def test_build_transaction_default_indent(lexicon):
    """_build_transaction combines assignment + identity with indent."""
    lexicon._current_type = "int"
    code = lexicon._build_transaction("v", 42, "  ")
    lines = code.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("  ")
    assert lines[1].startswith("  ")


def test_build_transaction_skip_first_indent(lexicon):
    """_build_transaction respects skip_first_indent flag."""
    lexicon._current_type = "int"
    code = lexicon._build_transaction("v", 42, "  ", skip_first_indent=True)
    lines = code.split("\n")
    assert not lines[0].startswith("  ")
    assert lines[1].startswith("  ")


# ===== Tests for Opaque Predicates / Loop Headers =====


def test_get_opaque_predicates(lexicon):
    """_get_opaque_predicates returns class-level list."""
    preds = lexicon._get_opaque_predicates()
    assert preds == ["false_condition"]


def test_get_unreachable_loop_headers(lexicon):
    """_get_unreachable_loop_headers returns class-level list."""
    loops = lexicon._get_unreachable_loop_headers()
    assert loops == ["for(int {var}=0;{var}<0;++{var})"]


# ===== Tests for Random Value Generation =====


def test_generate_random_value_type(lexicon):
    """generate_random_value returns value matching internal type."""
    for _ in range(10):
        val = lexicon.generate_random_value()
        assert lexicon._current_type in ["int", "float", "string"]
        if lexicon._current_type == "int":
            assert isinstance(val, int)
        elif lexicon._current_type == "float":
            assert isinstance(val, float)
        else:
            assert isinstance(val, str)


# ===== Tests for get_random_dead_code strategy delegation =====


def test_get_random_dead_code_assignment_strategy(monkeypatch, lexicon):
    """get_random_dead_code delegates to _build_transaction for assignment."""
    monkeypatch.setattr(
        lexicon._rng, "choice", lambda x: "assignment" if "assignment" in x else x[0]
    )
    code = lexicon.get_random_dead_code("v", "i", "  ")
    # Ensure that transaction body exists (delegation works)
    assert "v =" in code
    assert "v = v" in code


def test_get_random_dead_code_if_strategy(monkeypatch, lexicon):
    """get_random_dead_code delegates to _build_transaction + format_block for if_wrap."""

    def fake_choice(options):
        if "if_wrap" in options:
            return "if_wrap"
        return options[0]

    monkeypatch.setattr(lexicon._rng, "choice", fake_choice)
    code = lexicon.get_random_dead_code("v", "i", "  ")
    # Ensure header and body present
    assert "false_condition" in code
    assert "v = v" in code


def test_get_random_dead_code_loop_strategy(monkeypatch, lexicon):
    """get_random_dead_code delegates to _build_transaction + format_block for loop_wrap."""

    def fake_choice(options):
        if "loop_wrap" in options:
            return "loop_wrap"
        elif options == lexicon.UNREACHABLE_LOOP_HEADERS:
            return options[0]
        return options[0]

    monkeypatch.setattr(lexicon._rng, "choice", fake_choice)
    code = lexicon.get_random_dead_code("v", "loop_var", "  ")
    # Ensure header and body present
    assert "loop_var" in code
    assert "v = v" in code
