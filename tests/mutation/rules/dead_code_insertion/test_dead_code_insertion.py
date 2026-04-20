import pytest
import random
from unittest.mock import MagicMock, patch

from transtructiver.node import Node
from transtructiver.mutation.mutation_context import MutationContext
from transtructiver.mutation.rules.utils.scope_manager import ScopeManager
from transtructiver.mutation.rules.dead_code_insertion.dead_code_insertion import (
    DeadCodeInsertionRule,
)

# ===== Tree Construction Helpers =====


def _build_test_tree(with_whitespace=False):
    """
    Creates a valid, non-cyclic tree with parent links and a root scope.
    Optionally adds a whitespace node for testing indentation detection.
    Structure: Root(module) -> Func(function_definition) -> Block(block_scope) -> Stmt
    """
    # Create nodes
    stmt = Node((1, 4), (1, 8), "expression_statement", text="pass")
    block = Node((1, 0), (2, 0), "block", children=[stmt])
    func = Node((0, 0), (2, 0), "function_definition", children=[block])
    root = Node((0, 0), (2, 0), "module", children=[func])

    # Optionally add whitespace node
    if with_whitespace:
        ws_node = Node((2, 0), (2, 4), "whitespace", text="    ")  # column 0, 4 spaces
        root.children.append(ws_node)
        ws_node.parent = root

    # Set Metadata
    root.language = "python"
    root.semantic_label = "root"  # Triggers first enter_scope()
    func.semantic_label = "function_scope"
    block.semantic_label = "block_scope"  # Triggers _is_valid_container

    # Establish Parent Links (Crucial for _inject_dead_code)
    func.parent = root
    block.parent = func
    stmt.parent = block

    return root, block, stmt


# ===== Fixtures =====


@pytest.fixture
def mutation_context():
    return MutationContext()


@pytest.fixture
def mock_registry():
    """Mocks both lexicon and strategy registries."""
    with patch(
        "transtructiver.mutation.rules.dead_code_insertion.dead_code_insertion.get_lexicon"
    ) as m_lex, patch(
        "transtructiver.mutation.rules.dead_code_insertion.dead_code_insertion.get_strategy"
    ) as m_strat:

        # Lexicon Setup
        lex_inst = MagicMock()
        lex_inst.get_random_dead_code.return_value = "if False: pass\n"
        m_lex.return_value = lambda rng: lex_inst

        # Strategy Setup
        strat_inst = MagicMock()
        strat_inst.is_valid_container.return_value = True
        strat_inst.is_valid_gap.return_value = True
        strat_inst.get_indent_prefix.return_value = "    "
        strat_inst.is_terminal.return_value = False
        m_strat.return_value = strat_inst

        yield lex_inst, strat_inst


# ===== Tests =====


class TestDeadCodeInsertionRule:

    def test_init_method_sets_attributes(self):
        """Init sets level, RNG, base_indent, and scope manager correctly."""
        indent = "  "
        seed = 123
        rule = DeadCodeInsertionRule(level=2, seed=seed, indent_unit=indent)

        assert rule._level == 2
        other_rng = random.Random(seed)
        assert rule._rng.random() == other_rng.random()
        assert rule._base_indent == indent
        assert isinstance(rule._scope, ScopeManager)

    def test_apply_probability_logic(self, mutation_context, mock_registry):
        """Tests that the rule respects the inverted probability check (> prob)."""
        lex, strat = mock_registry
        root, block, stmt = _build_test_tree()

        # Level 3 -> Prob = 0.5. rng.random() is mocked to 0.1
        # 0.1 > 0.5 is False. It should NOT inject during scan, but hit fallback.
        rule = DeadCodeInsertionRule(level=3)
        with patch.object(rule._rng, "random", return_value=0.1):
            records = rule.apply(root, mutation_context)

        assert len(records) == 1
        # Verify fallback was used (records comes from _ensure_minimum_mutation)
        assert any(n.type == "dead_code" for n in block.children)

    def test_synthetic_id_decrement(self, mutation_context, mock_registry):
        """Verifies injected nodes pull unique IDs from the context."""
        root, block, _ = _build_test_tree()
        rule = DeadCodeInsertionRule()

        initial_id = mutation_context.synthetic_row_counter  # -1
        rule.apply(root, mutation_context)

        # Find the injected node
        dc_node = [n for n in block.children if n.type == "dead_code"][0]
        assert dc_node.start_point[0] == initial_id
        # Context should have decremented
        assert mutation_context.synthetic_row_counter == initial_id - 1

    def test_terminal_node_truncation(self, mutation_context, mock_registry):
        """Strategy.is_terminal should stop the injection scan for that block."""
        lex, strat = mock_registry
        root, block, stmt = _build_test_tree()

        # Add a second statement
        stmt2 = Node((2, 4), (2, 8), "expression_statement", text="pass")
        block.children.append(stmt2)

        # Mock strategy to say the first statement is terminal (like a return)
        strat.is_terminal.side_effect = lambda n: n == stmt

        rule = DeadCodeInsertionRule(level=3)
        # Force high probability of injection
        with patch.object(rule._rng, "random", return_value=0.9):
            rule.apply(root, mutation_context)

        # Injection should only be possible before the terminal node
        # Because the loop breaks, candidates should only contain the first gap
        assert len(block.children) == 3  # original 2 + 1 injected
        assert block.children[0].type == "dead_code"
        assert block.children[1] == stmt
