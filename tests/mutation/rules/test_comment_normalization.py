"""Unit tests for the CommentNormalizationRule mutation rule."""

import pytest
from transtructiver.mutation.rules.comment_normalization.comment_normalization import (
    CommentNormalizationRule,
)
from transtructiver.node import Node
from transtructiver.mutation.mutation_context import MutationContext
from transtructiver.mutation.rules.mutation_rule import MutationRecord

# ===== Helpers =====


def _wire_parents(node: Node, parent: Node | None = None) -> Node:
    """Recursively populate parent links after literal tree construction."""
    node.parent = parent
    for child in node.children:
        _wire_parents(child, node)
    return node


def label_nodes(root: Node) -> Node:
    """Annotate semantic labels and fields for comment nodes."""
    _wire_parents(root)
    root.semantic_label = "root"
    if not getattr(root, "language", None):
        root.language = "java"
    for node in root.traverse():
        if node.type == "method_declaration":
            node.semantic_label = "function_scope"
        elif node.type == "class_declaration":
            node.semantic_label = "class_scope"
        elif node.type == "block":
            node.semantic_label = "block_scope"
        elif node.type == "for_statement":
            node.semantic_label = "loop_scope"
        elif node.type == "if_statement":
            node.semantic_label = "condition_scope"
        elif node.type == "line_comment":
            node.semantic_label = "line_comment"
        elif node.type == "block_comment":
            node.semantic_label = "block_comment"
    return root


def make_line_comment_tree(length: int = 1, comment_text="// comment") -> Node:
    """Create a root node with a single line comment."""
    text = comment_text
    if length > 3:
        text = "// this comment has five words"
    if length > 8:
        text = "// this is a comment with more than eight words"
    comment = Node((1, 0), (1, len(text)), "line_comment", text=text)
    root = Node((0, 0), (2, 0), "program", children=[comment])
    return label_nodes(root)


def make_block_comment_tree(length: int = 1, comment_text="/* comment */") -> Node:
    """Create a root node with a single block comment."""
    text = comment_text
    if length > 3:
        text = "/* this comment has five words */"
    if length > 8:
        text = "/* this is a comment with more than eight words */"
    comment = Node((1, 0), (1, len(text)), "block_comment", text=text)
    root = Node((0, 0), (2, 0), "program", children=[comment])
    return label_nodes(root)


def make_python_docstring_tree(length: int = 1) -> Node:
    """Create a root node with a Python triple-quoted docstring."""
    text = "''' comment '''"
    if length > 3:
        text = "''' this comment has five words '''"
    if length > 8:
        text = "''' this is a comment with more than eight words '''"
    comment = Node((1, 0), (1, len(text)), "block_comment", text=text)
    root = Node((0, 0), (2, 0), "module", children=[comment])
    root.language = "python"
    return label_nodes(root)


def make_comment_in_scope_tree(
    scope_type="method_declaration", comment_type="line_comment", text="// comment"
) -> Node:
    """Create a scope node (function/class/loop/condition) with a comment child."""
    comment = Node((2, 0), (2, len(text)), comment_type, text=text)
    scope = Node((1, 0), (3, 0), scope_type, children=[comment])
    root = Node((0, 0), (4, 0), "program", children=[scope])
    return label_nodes(root)


def make_comment_after_terminal_tree() -> Node:
    """Create a block with a return statement followed by a comment."""
    return_stmt = Node((1, 0), (1, 6), "return_statement", text="return")
    comment = Node((2, 0), (2, 10), "line_comment", text="// after return")
    block = Node((0, 0), (3, 0), "block", children=[return_stmt, comment])
    return label_nodes(block)


def make_multiple_comments_tree() -> Node:
    """Create a root node with multiple comments."""
    c1 = Node((1, 0), (1, 8), "line_comment", text="// foo")
    c2 = Node((2, 0), (2, 8), "block_comment", text="/* bar */")
    root = Node((0, 0), (3, 0), "program", children=[c1, c2])
    return label_nodes(root)


def make_empty_tree() -> Node:
    """Create an empty root node."""
    root = Node((0, 0), (0, 0), "module", children=[])
    root.language = "python"
    return root


# ===== Fixtures =====


@pytest.fixture
def mutation_context():
    """Return a default MutationContext for testing."""
    return MutationContext()


@pytest.fixture
def line_comment_tree():
    """Returns a program root with a single line comment."""
    return make_line_comment_tree()


@pytest.fixture
def block_comment_tree():
    """Returns a program root with a single block comment."""
    return make_block_comment_tree()


@pytest.fixture
def python_docstring_tree():
    """Returns a module root with a single Python docstring."""
    return make_python_docstring_tree()


@pytest.fixture
def function_scope_comment_tree():
    """Returns a method_declaration node with a line comment child."""
    return make_comment_in_scope_tree("method_declaration", "line_comment", "// in function")


@pytest.fixture
def class_scope_comment_tree():
    """Returns a class_declaration node with a line comment child."""
    return make_comment_in_scope_tree("class_declaration", "line_comment", "// in class")


@pytest.fixture
def loop_scope_comment_tree():
    """Returns a for_statement node with a line comment child."""
    return make_comment_in_scope_tree("for_statement", "line_comment", "// in loop")


@pytest.fixture
def condition_scope_comment_tree():
    """Returns an if_statement node with a line comment child."""
    return make_comment_in_scope_tree("if_statement", "line_comment", "// in condition")


@pytest.fixture
def comment_after_terminal_tree():
    """Returns a block node with a return statement and a line comment."""
    return make_comment_after_terminal_tree()


@pytest.fixture
def multiple_comments_tree():
    """Returns a program root with multiple comments."""
    return make_multiple_comments_tree()


@pytest.fixture
def empty_tree():
    """Returns an empty module root node."""
    return make_empty_tree()


# ===== Test Cases =====


class TestCommentNormalizationCoreBehavior:
    """Core initialization, mutation, and record behavior."""

    def test_rule_initialization(self):
        """Ensure the rule initializes base and custom fields correctly."""
        rule = CommentNormalizationRule()
        assert rule.rule_name == "comment-normalization"

    def test_apply_with_none_root(self):
        """Ensure apply handles None root safely."""
        rule = CommentNormalizationRule()
        assert rule.apply(None, mutation_context) == []  # type: ignore

    @pytest.mark.parametrize("length", [1, 4, 9])
    def test_normalizes_line_comment(self, length, mutation_context):
        """It normalizes a single line comment and preserves delimiters."""
        tree = make_line_comment_tree(length)
        comment = tree.children[0]
        comment_text = comment.text
        rule = CommentNormalizationRule()
        records = rule.apply(tree, mutation_context)
        assert comment.text.startswith("//")  # type: ignore
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text

    @pytest.mark.parametrize("length", [1, 4, 9])
    def test_normalizes_block_comment(self, length, mutation_context):
        """It normalizes a block comment and preserves delimiters."""
        tree = make_block_comment_tree(length)
        comment = tree.children[0]
        comment_text = comment.text
        rule = CommentNormalizationRule()
        records = rule.apply(tree, mutation_context)
        assert comment.text.startswith("/*")  # type: ignore
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text

    @pytest.mark.parametrize("length", [1, 4, 9])
    def test_normalizes_python_docstring(self, length, mutation_context):
        """It normalizes Python triple-quoted docstrings."""
        tree = make_python_docstring_tree(length)
        comment = tree.children[0]
        comment_text = comment.text
        rule = CommentNormalizationRule()
        records = rule.apply(tree, mutation_context)
        assert comment.text.startswith("'''")  # type: ignore
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text

    def test_multiple_comments(self, multiple_comments_tree, mutation_context):
        """It normalizes multiple comments in the same tree."""
        comment_one_text = multiple_comments_tree.children[0].text
        comment_two_text = multiple_comments_tree.children[1].text
        rule = CommentNormalizationRule()
        records = rule.apply(multiple_comments_tree, mutation_context)
        assert len(records) == 2
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_one_text
        assert "new_val" in records[1].metadata
        assert records[1].metadata.get("new_val") != comment_two_text

    def test_empty_tree_returns_empty_list(self, empty_tree, mutation_context):
        """It returns an empty list when applied to an empty tree."""
        rule = CommentNormalizationRule()
        assert rule.apply(empty_tree, mutation_context) == []


class TestCommentNormalizationScopeAndContext:
    """Scope/context handling and edge case behavior."""

    def test_comment_in_function_scope(self, function_scope_comment_tree, mutation_context):
        """It normalizes comments inside a function scope."""
        comment_text = function_scope_comment_tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(function_scope_comment_tree, mutation_context)
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text

    def test_comment_in_class_scope(self, class_scope_comment_tree, mutation_context):
        """It normalizes comments inside a class scope."""
        comment_text = class_scope_comment_tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(class_scope_comment_tree, mutation_context)
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text

    def test_comment_in_loop_scope(self, loop_scope_comment_tree, mutation_context):
        """It normalizes comments inside a loop scope."""
        comment_text = loop_scope_comment_tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(loop_scope_comment_tree, mutation_context)
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text

    def test_comment_in_condition_scope(self, condition_scope_comment_tree, mutation_context):
        """It normalizes comments inside a condition scope."""
        comment_text = condition_scope_comment_tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(condition_scope_comment_tree, mutation_context)
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text

    def test_comment_after_terminal_statement(self, comment_after_terminal_tree, mutation_context):
        """It normalizes comments after terminal statements (return, break, etc.)."""
        comment_text = comment_after_terminal_tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(comment_after_terminal_tree, mutation_context)
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text


class TestCommentNormalizationFormatting:
    """Delimiter preservation and formatting behavior."""

    def test_comment_with_only_delimiters(self, mutation_context):
        """It handles comments that contain only delimiters."""
        tree = make_line_comment_tree(comment_text="//")
        comment_text = tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(tree, mutation_context)
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text

    def test_comment_with_whitespace(self, mutation_context):
        """It normalizes comments with leading/trailing whitespace."""
        tree = make_line_comment_tree(comment_text="   // comment   ")
        comment_text = tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(tree, mutation_context)
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text

    def test_comment_with_special_characters(self, mutation_context):
        """It normalizes comments containing special characters or unicode."""
        tree = make_line_comment_tree(comment_text="// cömment 🚀")
        comment_text = tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(tree, mutation_context)
        assert len(records) == 1
        assert "new_val" in records[0].metadata
        assert records[0].metadata.get("new_val") != comment_text


class TestCommentNormalizationErrorHandling:
    """Error and edge case handling."""

    def test_comment_with_no_parent_raises(self, line_comment_tree, mutation_context):
        """It raises ValueError if a comment node has no parent or scope."""
        comment = line_comment_tree.children[0]
        # Remove parent reference
        comment.parent = None
        rule = CommentNormalizationRule()
        with pytest.raises(ValueError):
            rule.apply(line_comment_tree, mutation_context)

    def test_root_node_with_no_language_raises(self, line_comment_tree, mutation_context):
        """It raises ValueError if the root node has no language set."""
        line_comment_tree.language = None
        rule = CommentNormalizationRule()
        with pytest.raises(ValueError):
            rule.apply(line_comment_tree, mutation_context)

    def test_unsupported_delimiter_is_unchanged(self, mutation_context):
        """It does not change comments with unknown or unsupported delimiters."""
        tree = make_line_comment_tree(comment_text="!! comment")
        tree.language = "python"
        rule = CommentNormalizationRule()
        records = rule.apply(tree, mutation_context)
        assert records == []


class TestCommentNormalizationMutationRecord:
    """MutationRecord content and correctness."""

    def test_mutation_record_content_line_comment(self, line_comment_tree, mutation_context):
        """It generates correct MutationRecord for normalized line comments."""
        comment = line_comment_tree.children[0]
        comment_text = line_comment_tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(line_comment_tree, mutation_context)
        assert records
        record = records[0]
        assert isinstance(record, MutationRecord)
        assert record.node_id == comment.start_point
        assert "new_val" in record.metadata
        assert record.metadata.get("new_val") != comment_text
        assert str(record.metadata.get("new_val")).startswith("// ")

    def test_mutation_record_content_block_comment(self, block_comment_tree, mutation_context):
        """It generates correct MutationRecord for normalized block comments."""
        comment = block_comment_tree.children[0]
        comment_text = block_comment_tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(block_comment_tree, mutation_context)
        assert records
        record = records[0]
        assert isinstance(record, MutationRecord)
        assert record.node_id == comment.start_point
        assert "new_val" in record.metadata
        assert record.metadata.get("new_val") != comment_text
        assert str(record.metadata.get("new_val")).startswith("/*\n")

    def test_mutation_record_content_doc_comment(self, mutation_context):
        """It generates correct MutationRecord for normalized block comments."""
        tree = make_block_comment_tree(comment_text="/** doc comment */")
        comment = tree.children[0]
        comment_text = tree.children[0].text
        rule = CommentNormalizationRule()
        records = rule.apply(tree, mutation_context)
        assert records
        record = records[0]
        assert isinstance(record, MutationRecord)
        assert record.node_id == comment.start_point
        assert "new_val" in record.metadata
        assert record.metadata.get("new_val") != comment_text
        assert str(record.metadata.get("new_val")).startswith("/**\n *")


@pytest.mark.parametrize("level", [0, 1, 2])
def test_rule_level_affects_replacement(line_comment_tree, level, mutation_context):
    """It uses the correct replacement strategy for different rule levels."""
    rule = CommentNormalizationRule(level=level)
    records = rule.apply(line_comment_tree, mutation_context)
    assert records
    assert len(records) > 0
