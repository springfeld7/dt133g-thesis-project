"""Comprehensive unit tests for identifier substitution renaming logic.

Tests cover:
  - Simple identifiers (1-3 words, no prepositions/suffixes)
  - Identifiers with single and multiple prepositions at various positions
  - Identifiers with suffix type replacements (str, list, map, set, num, etc.)
  - Combinations of prepositions and suffixes
  - Edge cases (empty, single word, all prepositions, all suffixes)
  - Case sensitivity (lowercase, UPPERCASE, camelCase, snake_case)
  - Deterministic behavior (fixed RNG seed)
"""

from unittest.mock import Mock
import pytest

from transtructiver.node import Node
from transtructiver.mutation.rules.identifier_renaming._rename_substitution import (
    _build_substitute_name,
    _SUFFIXES,
)


def mock_split_words(identifier):
    """Split on camelCase, snake_case, and whitespace."""
    parts = []
    current = []

    for i, char in enumerate(identifier):
        if char == "_" or char == "-" or char == " ":
            if current:
                parts.append("".join(current))
                current = []
        elif i > 0 and char.isupper() and identifier[i - 1].islower():
            if current:
                parts.append("".join(current))
                current = [char]
        else:
            current.append(char)

    if current:
        parts.append("".join(current))

    return parts


def mock_format_identifier(node, text, language):
    """Return text as-is to isolate rearrangement logic from formatting."""
    return text


@pytest.fixture
def mock_formatter(monkeypatch):
    """Patch external dependencies with mocks."""
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.split_words",
        mock_split_words,
    )
    monkeypatch.setattr(
        "transtructiver.mutation.rules.identifier_renaming._rename_substitution.format_identifier",
        mock_format_identifier,
    )


def create_node(text):
    """Create a mock Node with the given text."""
    node = Mock(spec=Node)
    node.text = text
    return node


# ==================== SIMPLE IDENTIFIERS (NO PREPOSITIONS/SUFFIXES) ====================

def test_single_word_identifier(mock_formatter):
    """Single-word identifiers should remain unchanged."""
    node = create_node("variable")
    result = _build_substitute_name(node, "python")
    assert result == "variable"


def test_two_word_identifier_no_preposition(mock_formatter):
    """Two-word identifier without prepositions should reverse."""
    node = create_node("coolAbility")
    result = _build_substitute_name(node, "python")
    assert result == "ability_cool"


def test_three_word_identifier_no_preposition(mock_formatter):
    """Three-word identifier should reverse to opposite order."""
    node = create_node("myAwesomeFunction")
    result = _build_substitute_name(node, "python")
    assert result == "function_awesome_my"


def test_four_word_identifier(mock_formatter):
    """Four-word identifier should reverse completely."""
    node = create_node("getCustomUserData")
    result = _build_substitute_name(node, "python")
    assert "data_user_custom_get" in result


# ==================== IDENTIFIERS WITH PREPOSITIONS ====================

def test_preposition_in_middle(mock_formatter):
    """Preposition in middle should now also be reversed with other words."""
    node = create_node("getDataFor")
    result = _build_substitute_name(node, "python")
    assert result == "for_data_get"


def test_preposition_for_in_middle(mock_formatter):
    """Test 'for' preposition specifically—now reversed with other words."""
    node = create_node("coolAbilityForCharStr")
    result = _build_substitute_name(node, "java")
    str_suffixes = _SUFFIXES["str"]
    has_str_suffix = any(suffix in result for suffix in str_suffixes)
    assert "for_char_ability_cool" in result
    assert has_str_suffix


def test_preposition_at_start(mock_formatter):
    """Preposition at the start should stay at start."""
    node = create_node("inTestVariable")
    result = _build_substitute_name(node, "python")
    assert result == "in_variable_test"


def test_multiple_prepositions(mock_formatter):
    """Multiple prepositions should now be reversed with other words."""
    node = create_node("getDataFromUserInDatabase")
    result = _build_substitute_name(node, "python")
    assert result == "in_database_from_user_data_get"


def test_consecutive_prepositions(mock_formatter):
    """Consecutive prepositions should now be reversed with other words."""
    node = create_node("moveFromToFor")
    result = _build_substitute_name(node, "python")
    assert result == "for_to_from_move"


# ==================== IDENTIFIERS WITH SUFFIX TYPES ====================

def test_list_suffix_replacement(mock_formatter):
    """Identifier ending with 'list' should replace with list suffixes."""
    node = create_node("userDataList")
    result = _build_substitute_name(node, "python")
    list_suffixes = _SUFFIXES["list"]
    has_list_suffix = any(suffix in result for suffix in list_suffixes)
    assert has_list_suffix


def test_str_suffix_replacement(mock_formatter):
    """Identifier ending with 'str' should replace with str suffixes."""
    node = create_node("characterStr")
    result = _build_substitute_name(node, "python")
    str_suffixes = _SUFFIXES["str"]
    has_str_suffix = any(suffix in result for suffix in str_suffixes)
    assert has_str_suffix


def test_map_suffix_replacement(mock_formatter):
    """Identifier ending with 'map' should use map suffixes."""
    node = create_node("configMap")
    result = _build_substitute_name(node, "python")
    map_suffixes = _SUFFIXES["map"]
    has_map_suffix = any(suffix in result for suffix in map_suffixes)
    assert has_map_suffix


def test_set_suffix_replacement(mock_formatter):
    """Identifier ending with 'set' should use set suffixes."""
    node = create_node("uniqueItemSet")
    result = _build_substitute_name(node, "python")
    set_suffixes = _SUFFIXES["set"]
    has_set_suffix = any(suffix in result for suffix in set_suffixes)
    assert has_set_suffix


def test_num_suffix_replacement(mock_formatter):
    """Identifier ending with 'num' should use num suffixes."""
    node = create_node("maxNum")
    result = _build_substitute_name(node, "python")
    num_suffixes = _SUFFIXES["num"]
    has_num_suffix = any(suffix in result for suffix in num_suffixes)
    assert has_num_suffix


def test_func_suffix_replacement(mock_formatter):
    """Identifier ending with 'func' should use func suffixes."""
    node = create_node("callbackFunc")
    result = _build_substitute_name(node, "python")
    func_suffixes = _SUFFIXES["func"]
    has_func_suffix = any(suffix in result for suffix in func_suffixes)
    assert has_func_suffix


def test_cls_suffix_replacement(mock_formatter):
    """Identifier ending with 'cls' should use cls suffixes."""
    node = create_node("componentCls")
    result = _build_substitute_name(node, "python")
    cls_suffixes = _SUFFIXES["cls"]
    has_cls_suffix = any(suffix in result for suffix in cls_suffixes)
    assert has_cls_suffix


# ==================== COMBINED: PREPOSITIONS + SUFFIXES ====================

def test_preposition_and_suffix_combined(mock_formatter):
    """Identifier with both prepositions and suffix should handle both correctly."""
    node = create_node("coolAbilityForCharStr")
    result = _build_substitute_name(node, "python")
    assert "for_char_ability_cool" in result
    str_suffixes = _SUFFIXES["str"]
    has_str_suffix = any(suffix in result for suffix in str_suffixes)
    assert has_str_suffix


def test_multiple_prepositions_with_suffix(mock_formatter):
    """Complex case: multiple prepositions + suffix at end."""
    node = create_node("getDataFromUserInList")
    result = _build_substitute_name(node, "python")
    assert "in_from_user_data_get_array" in result
    list_suffixes = _SUFFIXES["list"]
    has_list_suffix = any(suffix in result for suffix in list_suffixes)
    assert has_list_suffix


# ==================== EDGE CASES ====================

def test_empty_identifier(mock_formatter):
    """Empty identifier should return empty string."""
    node = create_node("")
    result = _build_substitute_name(node, "python")
    assert result == ""


def test_none_text(mock_formatter):
    """Node with None text should return empty string."""
    node = create_node(None)
    result = _build_substitute_name(node, "python")
    assert result == ""


def test_only_preposition(mock_formatter):
    """Identifier that is only a preposition."""
    node = create_node("for")
    result = _build_substitute_name(node, "python")
    assert result == "for"


def test_only_suffix(mock_formatter):
    """Identifier that is only a suffix type."""
    node = create_node("list")
    result = _build_substitute_name(node, "python")
    list_suffixes = _SUFFIXES["list"]
    has_list_suffix = any(suffix in result for suffix in list_suffixes)
    assert has_list_suffix, print(f"Result {result} should have list suffix")


def test_mixed_case_preposition_matching(mock_formatter):
    """Prepositions should match case-insensitively and reverse with other words."""
    node = create_node("myDataFOR")
    result = _build_substitute_name(node, "python")
    assert result == "for_data_my"


def test_mixed_case_suffix_matching(mock_formatter):
    """Suffixes should match case-insensitively."""
    node = create_node("myDataSTR")
    result = _build_substitute_name(node, "python")
    str_suffixes = _SUFFIXES["str"]
    has_str_suffix = any(suffix in result for suffix in str_suffixes)
    assert has_str_suffix


def test_five_word_identifier_no_special(mock_formatter):
    """Longer identifier without prepositions/suffixes should fully reverse."""
    node = create_node("myVeryAwesomeCustomFunction")
    result = _build_substitute_name(node, "python")
    assert result == "function_custom_awesome_very_my"


def test_deterministic_suffix_selection(mock_formatter):
    """Same identifier should always produce same suffix replacement (seed=42)."""
    node1 = create_node("dataStr")
    result1 = _build_substitute_name(node1, "python")

    node2 = create_node("dataStr")
    result2 = _build_substitute_name(node2, "python")

    assert result1 == result2


# ==================== REALISTIC IDENTIFIER PATTERNS ====================

def test_getter_method_pattern(mock_formatter):
    """Realistic getter method: getCustomUserData."""
    node = create_node("getCustomUserData")
    result = _build_substitute_name(node, "java")
    assert result == "data_user_custom_get"


def test_setter_method_pattern(mock_formatter):
    """Realistic setter method: setValueFor."""
    node = create_node("setValueFor")
    result = _build_substitute_name(node, "java")
    assert result == "for_value_set"


def test_database_query_pattern(mock_formatter):
    """Realistic DB pattern: selectUsersFromDatabase."""
    node = create_node("selectUsersFromDatabase")
    result = _build_substitute_name(node, "python")
    assert result == "from_database_users_select"


def test_list_variable_pattern(mock_formatter):
    """Realistic list: activeUsersList."""
    node = create_node("activeUsersList")
    result = _build_substitute_name(node, "python")
    list_suffixes = _SUFFIXES["list"]
    has_list_suffix = any(suffix in result for suffix in list_suffixes)
    assert has_list_suffix


def test_map_variable_pattern(mock_formatter):
    """Realistic map: userNamesMap."""
    node = create_node("userNamesMap")
    result = _build_substitute_name(node, "python")
    map_suffixes = _SUFFIXES["map"]
    has_map_suffix = any(suffix in result for suffix in map_suffixes)
    assert has_map_suffix


def test_configuration_pattern(mock_formatter):
    """Realistic config: loadConfigFrom."""
    node = create_node("loadConfigFrom")
    result = _build_substitute_name(node, "python")
    assert result == "from_config_load"


def test_event_handler_pattern(mock_formatter):
    """Realistic event handler: onClickHandlerForButton."""
    node = create_node("onClickHandlerForButton")
    result = _build_substitute_name(node, "java")
    assert result == "for_button_on_handler_click"


def test_iterator_pattern(mock_formatter):
    """Realistic iterator: iterateOverItemsFunc."""
    node = create_node("iterateOverItemsFunc")
    result = _build_substitute_name(node, "python")
    func_suffixes = _SUFFIXES["func"]
    has_func_suffix = any(suffix in result for suffix in func_suffixes)
    assert has_func_suffix


def test_validation_pattern(mock_formatter):
    """Realistic validation: isValidUserInputFor."""
    node = create_node("isValidUserInputFor")
    result = _build_substitute_name(node, "java")
    assert result == "for_input_user_valid_is"


def test_nested_structure_pattern(mock_formatter):
    """Realistic nested structure: getUsersFromDatabaseMap."""
    node = create_node("getUsersFromDatabaseMap")
    result = _build_substitute_name(node, "python")
    map_suffixes = _SUFFIXES["map"]
    has_map_suffix = any(suffix in result for suffix in map_suffixes)
    assert has_map_suffix


# ==================== MULTIPLE LANGUAGES ====================

def test_different_language_java(mock_formatter):
    """Should work correctly for Java identifiers."""
    node = create_node("myDataFor")
    result_java = _build_substitute_name(node, "java")
    assert result_java is not None
    assert "for" in result_java


def test_different_language_cpp(mock_formatter):
    """Should work correctly for C++ identifiers."""
    node = create_node("getUserDataList")
    result_cpp = _build_substitute_name(node, "cpp")
    list_suffixes = _SUFFIXES["list"]
    has_list_suffix = any(suffix in result_cpp for suffix in list_suffixes)
    assert has_list_suffix


def test_different_language_python(mock_formatter):
    """Should work correctly for Python identifiers."""
    node = create_node("process_user_data_from")
    result_py = _build_substitute_name(node, "python")
    assert result_py is not None


# ==================== SPECIAL PATTERNS ====================

@pytest.mark.parametrize("identifier,preposition", [
    ("getDataAt", "at"),
    ("moveItemIn", "in"),
    ("searchWithin", "within"),
    ("processBy", "by"),
    ("queryWith", "with"),
    ("transitionBetween", "between"),
])
def test_all_valid_prepositions(mock_formatter, identifier, preposition):
    """Test sampling of different prepositions are included in reversal."""
    node = create_node(identifier)
    result = _build_substitute_name(node, "python")
    assert preposition in result


@pytest.mark.parametrize("identifier,suffix_type", [
    ("dataList", "list"),
    ("dataTuple", "tuple"),
    ("dataMap", "map"),
    ("dataSet", "set"),
    ("valueStr", "str"),
    ("countNum", "num"),
    ("isFlag", "flag"),
    ("doFunc", "func"),
    ("MyClassCls", "cls"),
    ("fieldAttr", "attr"),
    ("pointerVar", "var"),
    ("inputParam", "param"),
])
def test_all_valid_suffixes(mock_formatter, identifier, suffix_type):
    """Test sampling of different suffix types are replaced."""
    node = create_node(identifier)
    result = _build_substitute_name(node, "python")
    expected_suffixes = _SUFFIXES[suffix_type]
    has_suffix = any(s in result for s in expected_suffixes)
    assert has_suffix, \
        f"{identifier} should have a {suffix_type} suffix replacement in {result}"


def test_arg_suffix_replacement(mock_formatter):
    """Identifier ending with 'arg' should use arg suffixes."""
    node = create_node("functionArg")
    result = _build_substitute_name(node, "python")
    arg_suffixes = _SUFFIXES["arg"]
    has_arg_suffix = any(suffix in result for suffix in arg_suffixes)
    assert has_arg_suffix
