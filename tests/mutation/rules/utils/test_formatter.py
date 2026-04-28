"""Tests for identifier formatting helpers."""

from unittest.mock import Mock
from transtructiver.node import Node
from transtructiver.mutation.rules.utils import formatter


class TestSplitWords:
    """Tests for split_words function."""

    def test_empty_text(self):
        """Test split_words returns empty list for empty text."""
        assert formatter.split_words("") == []

    def test_single_lowercase_word(self):
        """Test single lowercase word is returned as-is."""
        assert formatter.split_words("variable") == ["variable"]

    def test_single_uppercase_word(self):
        """Test single uppercase word is returned as-is."""
        assert formatter.split_words("VARIABLE") == ["VARIABLE"]

    def test_simple_camel_case(self):
        """Test camelCase splitting."""
        assert formatter.split_words("myVariable") == ["my", "Variable"]

    def test_pascal_case(self):
        """Test PascalCase splitting."""
        assert formatter.split_words("MyVariable") == ["My", "Variable"]

    def test_simple_snake_case(self):
        """Test snake_case splitting."""
        assert formatter.split_words("my_variable") == ["my", "variable"]

    def test_mixed_snake_and_camel_case(self):
        """Test mixed snake_case and camelCase splitting."""
        assert formatter.split_words("my_variableName") == ["my", "variable", "Name"]

    def test_consecutive_underscores(self):
        """Test consecutive underscores are treated as word boundaries."""
        assert formatter.split_words("my__variable") == ["my", "variable"]

    def test_leading_underscore(self):
        """Test leading underscore is consumed at boundary."""
        assert formatter.split_words("_myVariable") == ["my", "Variable"]

    def test_trailing_underscore(self):
        """Test trailing underscore is ignored."""
        assert formatter.split_words("myVariable_") == ["my", "Variable"]

    def test_consecutive_capitals(self):
        """Test consecutive capitals (XMLParser) - each stays together."""
        assert formatter.split_words("XMLParser") == ["XMLParser"]

    def test_capitals_followed_by_lowercase(self):
        """Test capitals followed by lowercase (XMLHttpRequest)."""
        assert formatter.split_words("XMLHttpRequest") == ["XMLHttp", "Request"]

    def test_single_character_words(self):
        """Test single character words."""
        assert formatter.split_words("a") == ["a"]

    def test_multiple_single_character_words(self):
        """Test multiple single character camelCase words."""
        assert formatter.split_words("aB") == ["a", "B"]

    def test_numbers_in_identifier(self):
        """Test identifiers with numbers don't split on digits."""
        # Numbers don't trigger camelCase boundaries (uppercase after lowercase)
        assert formatter.split_words("var2Name") == ["var2Name"]

    def test_numbers_and_underscore(self):
        """Test identifiers with numbers and underscores."""
        assert formatter.split_words("var2_name") == ["var2", "name"]

    def test_only_underscores(self):
        """Test string with only underscores."""
        assert formatter.split_words("___") == []


class TestFormatSnakeCase:
    """Tests for _format_snake_case function."""

    def test_empty_words_list(self):
        """Test formatting empty words list."""
        assert formatter._format_snake_case([]) == ""

    def test_single_word(self):
        """Test formatting single word."""
        assert formatter._format_snake_case(["MyWord"]) == "myword"

    def test_multiple_words(self):
        """Test formatting multiple words."""
        assert formatter._format_snake_case(["My", "Variable", "Name"]) == "my_variable_name"

    def test_single_character_words(self):
        """Test formatting when all words are single character."""
        assert formatter._format_snake_case(["a", "b", "c"]) == "abc"

    def test_mixed_single_and_multi_char_words(self):
        """Test formatting mixed single and multi-character words."""
        assert formatter._format_snake_case(["a", "Variable"]) == "a_variable"

    def test_already_lowercase(self):
        """Test formatting already lowercase words."""
        assert formatter._format_snake_case(["my", "variable"]) == "my_variable"

    def test_uppercase_words(self):
        """Test formatting uppercase words."""
        assert formatter._format_snake_case(["MY", "VARIABLE"]) == "my_variable"


class TestFormatCamelCase:
    """Tests for _format_camel_case function."""

    def test_empty_words_list(self):
        """Test formatting empty words list."""
        assert formatter._format_camel_case([]) == ""

    def test_single_word(self):
        """Test formatting single word to camelCase."""
        assert formatter._format_camel_case(["MyWord"]) == "myword"

    def test_multiple_words(self):
        """Test formatting multiple words to camelCase."""
        assert formatter._format_camel_case(["My", "Variable", "Name"]) == "myVariableName"

    def test_first_word_lowercase(self):
        """Test first word is lowercase, rest are capitalized."""
        assert formatter._format_camel_case(["my", "variable", "name"]) == "myVariableName"

    def test_single_character_words(self):
        """Test formatting single character words to camelCase."""
        assert formatter._format_camel_case(["a", "b", "c"]) == "aBC"

    def test_already_correct_format(self):
        """Test formatting already correct camelCase."""
        assert formatter._format_camel_case(["my", "Variable", "Name"]) == "myVariableName"


class TestFormatPascalCase:
    """Tests for _format_pascal_case function."""

    def test_empty_words_list(self):
        """Test formatting empty words list."""
        assert formatter._format_pascal_case([]) == ""

    def test_single_word(self):
        """Test formatting single word to PascalCase."""
        assert formatter._format_pascal_case(["myword"]) == "Myword"

    def test_multiple_words(self):
        """Test formatting multiple words to PascalCase."""
        assert formatter._format_pascal_case(["my", "variable", "name"]) == "MyVariableName"

    def test_already_uppercase(self):
        """Test formatting already uppercase words."""
        # capitalize() only affects the first character
        assert formatter._format_pascal_case(["MY", "VARIABLE"]) == "MyVariable"

    def test_single_character_words(self):
        """Test formatting single character words to PascalCase."""
        assert formatter._format_pascal_case(["a", "b", "c"]) == "ABC"

    def test_already_correct_format(self):
        """Test formatting already correct PascalCase."""
        assert formatter._format_pascal_case(["My", "Variable", "Name"]) == "MyVariableName"


class TestIsTitle:
    """Tests for _is_title function."""

    def test_python_class_name_is_title(self):
        """Test Python class_name semantic label is recognized as title."""
        node = Mock(spec=Node)
        node.semantic_label = "class_name"
        assert formatter._is_title(node, "python") is True

    def test_python_non_class_is_not_title(self):
        """Test Python non-class semantic label is not a title."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        assert formatter._is_title(node, "python") is False

    def test_java_class_name_is_title(self):
        """Test Java class_name semantic label is recognized as title."""
        node = Mock(spec=Node)
        node.semantic_label = "class_name"
        assert formatter._is_title(node, "java") is True

    def test_java_non_class_is_not_title(self):
        """Test Java non-class semantic label is not a title."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        assert formatter._is_title(node, "cpp") is False

    def test_cpp_class_name_is_title(self):
        """Test C++ class_name semantic label is recognized as title."""
        node = Mock(spec=Node)
        node.semantic_label = "class_name"
        assert formatter._is_title(node, "cpp") is True

    def test_cpp_non_class_is_not_title(self):
        """Test C++ non-class semantic label is not a title."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        assert formatter._is_title(node, "cpp") is False

    def test_unknown_language_is_not_title(self):
        """Test unknown language always returns False."""
        node = Mock(spec=Node)
        node.semantic_label = "class_name"
        assert formatter._is_title(node, "unknown_lang") is False


class TestFormatIdentifier:
    """Tests for format_identifier function."""

    def test_python_variable_name(self):
        """Test formatting Python variable name."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        result = formatter.format_identifier(node, "myVariableName", "python")
        assert result == "my_variable_name"

    def test_python_class_name(self):
        """Test formatting Python class name uses PascalCase."""
        node = Mock(spec=Node)
        node.semantic_label = "class_name"
        result = formatter.format_identifier(node, "myClassName", "python")
        assert result == "MyClassName"

    def test_java_variable_name(self):
        """Test formatting Java variable name uses camelCase."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        result = formatter.format_identifier(node, "MyVariableName", "java")
        assert result == "myVariableName"

    def test_java_class_name(self):
        """Test formatting Java class name uses PascalCase."""
        node = Mock(spec=Node)
        node.semantic_label = "class_name"
        result = formatter.format_identifier(node, "myClassName", "java")
        assert result == "MyClassName"

    def test_cpp_variable_name(self):
        """Test formatting C++ variable name uses camelCase."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        result = formatter.format_identifier(node, "MyVariableName", "cpp")
        assert result == "myVariableName"

    def test_destruct_title_python(self):
        """Test destruct prefix with title in Python."""
        node = Mock(spec=Node)
        node.semantic_label = "class_name"
        result = formatter.format_identifier(node, "destruct_c", "python")
        assert result == "C"

    def test_destruct_non_title_python(self):
        """Test destruct prefix without title in Python."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        result = formatter.format_identifier(node, "destruct_v", "python")
        assert result == "v"

    def test_destruct_title_java(self):
        """Test destruct prefix with title in Java."""
        node = Mock(spec=Node)
        node.semantic_label = "class_name"
        result = formatter.format_identifier(node, "destruct_c", "java")
        assert result == "C"

    def test_destruct_non_title_java(self):
        """Test destruct prefix without title in Java."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        result = formatter.format_identifier(node, "destruct_v", "java")
        assert result == "v"

    def test_snake_case_input_python(self):
        """Test formatting snake_case input for Python."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        result = formatter.format_identifier(node, "my_variable_name", "python")
        assert result == "my_variable_name"

    def test_empty_text(self):
        """Test formatting empty text."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        result = formatter.format_identifier(node, "", "python")
        assert result == ""

    def test_unknown_language_defaults_to_camel_case(self):
        """Test unknown language defaults to camelCase."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        result = formatter.format_identifier(node, "MyVariableName", "unknown_lang")
        assert result == "myVariableName"

    def test_single_word_python(self):
        """Test formatting single word for Python."""
        node = Mock(spec=Node)
        node.semantic_label = "variable_name"
        result = formatter.format_identifier(node, "variable", "python")
        assert result == "variable"

    def test_single_word_java_class(self):
        """Test formatting single word for Java class."""
        node = Mock(spec=Node)
        node.semantic_label = "class_name"
        result = formatter.format_identifier(node, "variable", "java")
        assert result == "Variable"
