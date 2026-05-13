"""Tests semantic identifier rename substitution output using mocked CST nodes."""

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*BPE.__init__ will not create from files anymore.*:DeprecationWarning"
)

from transtructiver.node import Node
from transtructiver.mutation.rules.identifier_renaming._rename_substitution import (
    _build_substitute_name,
)


class MockNode(Node):
    def __init__(self, text, semantic_label=None, parent=None):
        self._text = text
        self._semantic_label = semantic_label
        self.parent = parent
        self.children = []

    @property
    def text(self):
        return self._text

    @property
    def semantic_label(self):  # type: ignore
        return self._semantic_label

    def to_code(self):
        # For the test, we simulate the code block
        if "scope" in (self.semantic_label or ""):
            return self._text  # In this mock, 'text' holds the full scope code
        return self._text

    def traverse_up(self):
        curr = self
        while curr:
            yield curr
            curr = curr.parent


def test_rename_substitution_visual_check():
    """
    Visual verification of the MLM + VarCLR renaming logic.
    """
    test_cases = [
        {
            "original": "payload_arg",
            "code": "def send_packet(payload_arg, addr):\n    print(f'Sending {payload_arg} to {addr}')\n    return True",
            "lang": "python",
        },
        {
            "original": "radius_num",
            "code": "def calculate_area(radius_num):\n    import math\n    return math.pi * (radius_num ** 2)",
            "lang": "python",
        },
        {
            "original": "items_list",
            "code": "def process_data(items_list):\n    for i in range(len(items_list)):\n        print(items_list[i])",
            "lang": "python",
        },
        {
            "original": "db",
            "code": "class Database:\n    def connect(self, db_conn):\n        self.connection = db_conn.open()",
            "lang": "python",
        },
    ]

    print("\n" + "=" * 85)
    print(f"{'ORIGINAL':<20} -> {'RENAMED':<20} | {'CONTEXT'}")
    print("-" * 85)

    for case in test_cases:
        # Create a scope node (parent)
        scope_node = MockNode(case["code"], semantic_label="function_scope")

        # Create the target node (the variable to rename)
        target_node = MockNode(case["original"], parent=scope_node)

        # Run substitution
        renamed = _build_substitute_name(target_node, case["lang"])

        context_line = case["code"].split("\n")[0]
        print(f"{case['original']:<20} -> {renamed:<20} | {context_line}")

    print("=" * 85 + "\n")


# To run this specific test and see output:
# uv run pytest -s tests/mutation/rules/identifier_renaming/test_rename_substitution.py
