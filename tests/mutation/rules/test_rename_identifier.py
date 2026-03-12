"""Unit tests for the RenameIdentifiersRule mutation rule."""

import pytest
from transtructiver.mutation.rules.identifier_renaming.rename_identifiers import (
    RenameIdentifiersRule,
)
from transtructiver.node import Node
from transtructiver.mutation.rules.mutation_rule import MutationRecord
from transtructiver.mutation.mutation_types import MutationAction


@pytest.fixture
def sample_tree():
    """Create a sample node tree with identifiers for testing.

    Returns a root node with 4 children:
    - [0]: identifier "x"
    - [1]: identifier "y"
    - [2]: identifier "x"
    - [3]: number "1"
    """
    root = Node((0, 0), (0, 10), "module", children=[])
    root.semantic_label = "root"
    first_x = Node((1, 0), (1, 1), "identifier", text="x")
    first_y = Node((1, 2), (1, 3), "identifier", text="y")
    second_x = Node((1, 4), (1, 5), "identifier", text="x")
    non_identifier = Node((1, 6), (1, 7), "number", text="1")

    first_x.semantic_label = "variable_name"
    first_y.semantic_label = "variable_name"
    second_x.semantic_label = "variable_name"

    root.children = [first_x, first_y, second_x, non_identifier]

    return root


def test_rename_identifier_rule_initialization():
    """Ensure the subclass initializes base and custom fields correctly."""
    rule = RenameIdentifiersRule()

    assert rule.name == "RenameIdentifiersRule"


def test_rename_identifier_rule_apply_with_none_root():
    """Ensure apply handles None root safely."""
    rule = RenameIdentifiersRule()

    assert rule.apply(None) == []  # type: ignore


def test_rename_identifier_rule_mutates_node_text(sample_tree):
    """Ensure apply renames identifier node text values consistently."""
    first_x, first_y, second_x, non_identifier = sample_tree.children

    rule = RenameIdentifiersRule()
    rule.apply(sample_tree)

    assert first_x.text != "x"
    assert first_y.text != "y"
    assert second_x.text != "x"

    assert non_identifier.text == "1"
    assert second_x.text == first_x.text


def test_rename_identifier_rule_returns_mutation_records(sample_tree):
    """Ensure apply returns correct mutation records for renamed identifiers."""
    first_x, first_y, second_x, _ = sample_tree.children

    rule = RenameIdentifiersRule()
    records = rule.apply(sample_tree)

    assert len(records) == 3
    assert all(record.action == MutationAction.RENAME for record in records)
    assert records[0].node_id == (1, 0)

    assert records[0].metadata != {"new_val": "x"}
    assert records[0].metadata == {"new_val": first_x.text}

    assert records[1].metadata != {"new_val": "y"}
    assert records[1].metadata == {"new_val": first_y.text}

    assert records[2].metadata == {"new_val": second_x.text}
    assert records[2].metadata == {"new_val": first_x.text}


def _link(parent: Node, child: Node, field: str | None = None) -> None:
    """Attach child to parent and keep parent/field references consistent."""
    parent.children.append(child)
    child.parent = parent
    child.field = field


@pytest.fixture
def scoped_tree_with_shadowing() -> Node:
    """Create tree where inner scope declares the same identifier as outer scope."""
    root = Node((0, 0), (0, 0), "module", children=[])
    root.semantic_label = "root"

    outer_assignment = Node((1, 0), (1, 5), "assignment", children=[])
    outer_x = Node((1, 0), (1, 1), "identifier", text="x")
    outer_x.semantic_label = "variable_name"
    _link(outer_assignment, outer_x, field="left")
    _link(root, outer_assignment)

    function_def = Node((2, 0), (5, 0), "function_definition", children=[])
    function_def.semantic_label = "function_scope"
    function_name = Node((2, 4), (2, 7), "identifier", text="foo")
    function_name.semantic_label = "function_name"
    _link(function_def, function_name, field="name")

    function_block = Node((3, 0), (5, 0), "block", children=[])
    function_block.semantic_label = "block_scope"
    inner_assignment = Node((3, 4), (3, 9), "assignment", children=[])
    inner_x_decl = Node((3, 4), (3, 5), "identifier", text="x")
    inner_x_decl.semantic_label = "variable_name"
    _link(inner_assignment, inner_x_decl, field="left")

    inner_use = Node((4, 4), (4, 5), "identifier", text="x")
    inner_use.semantic_label = "variable_name"
    _link(function_block, inner_assignment)
    _link(function_block, inner_use)
    _link(function_def, function_block)
    _link(root, function_def)

    outer_use = Node((6, 0), (6, 1), "identifier", text="x")
    outer_use.semantic_label = "variable_name"
    _link(root, outer_use)

    return root


def test_rename_identifier_rule_uses_scope_stack_for_shadowing(scoped_tree_with_shadowing: Node):
    """Inner declaration of same name should not reuse outer scope rename."""
    rule = RenameIdentifiersRule()
    records: list[MutationRecord] = rule.apply(scoped_tree_with_shadowing)

    renamed_x_nodes: list[Node] = [
        node
        for node in scoped_tree_with_shadowing.traverse()
        if node.type == "identifier" and node.start_point in {(1, 0), (3, 4), (4, 4), (6, 0)}
    ]
    renamed_values: dict[tuple[int, int], str | None] = {
        node.start_point: node.text for node in renamed_x_nodes
    }

    assert len(records) == 4
    # Outer x and outer use should be equal
    assert renamed_values[(1, 0)] == renamed_values[(6, 0)]
    # Inner x and inner use should be equal
    assert renamed_values[(3, 4)] == renamed_values[(4, 4)]
    # Outer x and inner x should not be equal
    assert renamed_values[(1, 0)] != renamed_values[(3, 4)]


def test_rename_identifier_rule_clears_scope_state_between_runs(sample_tree):
    """Internal scope stack should be reset after apply finishes."""
    rule = RenameIdentifiersRule()
    rule.apply(sample_tree)

    assert rule.scope == []


def test_rename_identifier_rule_skips_unannotated_identifiers():
    """Identifiers without semantic labels should not be renamed."""
    root = Node((0, 0), (0, 3), "module", children=[])
    root.semantic_label = "root"
    ident = Node((1, 0), (1, 1), "identifier", text="x")
    root.children = [ident]

    rule = RenameIdentifiersRule()
    records = rule.apply(root)

    assert records == []
    assert ident.text == "x"


def test_rename_identifier_rule_skips_non_variable_annotations():
    """Only variable-like semantic labels should be renamed by default."""
    root = Node((0, 0), (0, 10), "module", children=[])
    root.semantic_label = "root"
    function_name = Node((1, 0), (1, 3), "identifier", text="foo")
    function_name.semantic_label = "function_name"
    root.children = [function_name]

    rule = RenameIdentifiersRule()
    records = rule.apply(root)

    assert records == []
    assert function_name.text == "foo"


def test_rename_identifier_rule_can_target_function_names_by_keyword():
    """Keyword-based targeting should allow function_name renames."""
    root = Node((0, 0), (0, 10), "module", children=[])
    root.semantic_label = "root"
    function_name = Node((1, 0), (1, 3), "identifier", text="foo")
    function_name.semantic_label = "function_name"
    root.children = [function_name]

    rule = RenameIdentifiersRule(targets=["function"])
    records = rule.apply(root)

    assert len(records) == 1
    assert function_name.text == "foo_fn"


def test_rename_identifier_rule_can_target_only_parameter_names():
    """Target keywords should restrict renaming to selected semantic labels."""
    root = Node((0, 0), (0, 10), "module", children=[])
    root.semantic_label = "root"

    variable_id = Node((1, 0), (1, 1), "identifier", text="x")
    variable_id.semantic_label = "variable_name"
    parameter_id = Node((1, 2), (1, 3), "identifier", text="p")
    parameter_id.semantic_label = "parameter_name"
    root.children = [variable_id, parameter_id]

    rule = RenameIdentifiersRule(targets=["parameter"])
    records = rule.apply(root)

    assert len(records) == 1
    assert variable_id.text == "x"
    assert parameter_id.text == "pp"


def test_rename_identifier_rule_rejects_unknown_target_keyword():
    """Unknown keyword should raise a clear configuration error."""
    with pytest.raises(ValueError, match="Unsupported rename target keyword"):
        RenameIdentifiersRule(targets=["not_a_real_target"])


def test_rename_identifier_rule_uses_parameter_type_for_suffix():
    """Typed parameters should receive a type-aware suffix when possible."""
    root = Node((0, 0), (0, 10), "module", children=[])
    root.semantic_label = "root"

    function_def = Node((1, 0), (1, 10), "function_definition", children=[])
    function_def.semantic_label = "function_scope"
    params = Node((1, 4), (1, 10), "parameters", children=[])
    typed_parameter = Node((1, 5), (1, 10), "typed_parameter", children=[])
    parameter_id = Node((1, 5), (1, 9), "identifier", text="items")
    parameter_id.field = "name"
    parameter_id.semantic_label = "parameter_name"
    type_node = Node((1, 11), (1, 15), "type", children=[])
    type_node.field = "type"
    list_type = Node((1, 11), (1, 15), "identifier", text="list")
    list_type.semantic_label = "type_name"

    _link(type_node, list_type)
    _link(typed_parameter, parameter_id)
    _link(typed_parameter, type_node, field="type")
    _link(params, typed_parameter)
    _link(function_def, params)
    _link(root, function_def)

    rule = RenameIdentifiersRule(targets=["parameter"])
    records = rule.apply(root)

    assert len(records) == 1
    assert parameter_id.text == "items_list"


def test_rename_identifier_rule_uses_assignment_value_type_for_suffix():
    """Assignment declarations should infer suffixes from assigned values."""
    root = Node((0, 0), (0, 10), "module", children=[])
    root.semantic_label = "root"

    list_assignment = Node((1, 0), (1, 10), "assignment", children=[])
    values_id = Node((1, 0), (1, 6), "identifier", text="values")
    values_id.field = "left"
    values_id.semantic_label = "variable_name"
    equals_a = Node((1, 7), (1, 8), "operator", text="=")
    list_literal = Node((1, 9), (1, 11), "list", children=[])

    _link(list_assignment, values_id, field="left")
    _link(list_assignment, equals_a)
    _link(list_assignment, list_literal)

    string_assignment = Node((2, 0), (2, 12), "assignment", children=[])
    message_id = Node((2, 0), (2, 7), "identifier", text="message")
    message_id.field = "left"
    message_id.semantic_label = "variable_name"
    equals_b = Node((2, 8), (2, 9), "operator", text="=")
    string_literal = Node((2, 10), (2, 17), "string", text='"hello"')

    _link(string_assignment, message_id, field="left")
    _link(string_assignment, equals_b)
    _link(string_assignment, string_literal)

    _link(root, list_assignment)
    _link(root, string_assignment)

    rule = RenameIdentifiersRule(targets=["variable"])
    records = rule.apply(root)

    assert len(records) == 2
    assert values_id.text == "valuess"
    assert message_id.text == "messagee"


def test_rename_identifier_rule_uses_java_style_for_program_root():
    """Program roots should use non-Python naming style formatting."""
    root = Node((0, 0), (0, 10), "program", children=[])
    root.semantic_label = "root"

    function_name = Node((1, 0), (1, 3), "identifier", text="foo")
    function_name.semantic_label = "function_name"
    root.children = [function_name]

    rule = RenameIdentifiersRule(targets=["function"])
    records = rule.apply(root)

    assert len(records) == 1
    assert function_name.text == "fooFn"


def test_rename_identifier_rule_formats_python_and_java_differently():
    """Suffix formatting should differ between module and program roots."""
    python_root = Node((0, 0), (0, 10), "module", children=[])
    python_root.semantic_label = "root"
    python_function = Node((1, 0), (1, 3), "identifier", text="foo")
    python_function.semantic_label = "function_name"
    python_root.children = [python_function]

    java_root = Node((0, 0), (0, 10), "program", children=[])
    java_root.semantic_label = "root"
    java_function = Node((1, 0), (1, 3), "identifier", text="foo")
    java_function.semantic_label = "function_name"
    java_root.children = [java_function]

    python_rule = RenameIdentifiersRule(targets=["function"])
    java_rule = RenameIdentifiersRule(targets=["function"])
    python_rule.apply(python_root)
    java_rule.apply(java_root)

    assert python_function.text == "foo_fn"
    assert java_function.text == "fooFn"


def test_rename_identifier_rule_empty_suffix_fallback_is_stable():
    """When no suffix can be inferred, renaming should use fallback behavior."""
    root = Node((0, 0), (0, 10), "module", children=[])
    root.semantic_label = "root"

    variable_id = Node((1, 0), (1, 4), "identifier", text="data")
    variable_id.semantic_label = "variable_name"
    root.children = [variable_id]

    rule = RenameIdentifiersRule(targets=["variable"])
    records = rule.apply(root)

    assert len(records) == 1
    assert variable_id.text == "dataa"


def test_rename_identifier_rule_overwrites_language_each_apply_run():
    """Language chosen per root should not leak between consecutive apply calls."""
    python_root = Node((0, 0), (0, 10), "module", children=[])
    python_root.semantic_label = "root"
    python_function = Node((1, 0), (1, 3), "identifier", text="foo")
    python_function.semantic_label = "function_name"
    python_root.children = [python_function]

    java_root = Node((0, 0), (0, 10), "program", children=[])
    java_root.semantic_label = "root"
    java_function = Node((1, 0), (1, 3), "identifier", text="bar")
    java_function.semantic_label = "function_name"
    java_root.children = [java_function]

    rule = RenameIdentifiersRule(targets=["function"])
    rule.apply(python_root)
    rule.apply(java_root)

    assert python_function.text == "foo_fn"
    assert java_function.text == "barFn"
