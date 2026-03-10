"""Entry point for parsing demo.

Runs the parser on a small example and prints the resulting CST as a
pretty-printed Node tree.
"""

from collections import deque

from ..node import Node
from .parser import Parser


INTERESTING_IDENTIFIERS = [
    "variable_declarator/java",
    "declaration/cpp",
    "assignment_expression/cpp",
]


def main():
    """Run a simple parsing demo and print the CST."""
    parser = Parser()

    # Test examples for different languages
    test_cases = [
        (
            "Python",
            '''def check_grammar(text: str = ""):
    """
    Checks grammar and spelling in the given text using LanguageTool.
    Returns a list of issues found.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be a non-empty string.")

    # Create a grammar checking tool (default: English)
    tool = language_tool_python.LanguageTool('en-US')

    # Find matches (grammar/spelling issues)
    matches = tool.check(text)

    return matches

def suggest_corrections(text = None):
    """
    Prints grammar issues and suggested corrections.
    """
    matches = check_grammar(text)

    if not matches:
        print("✅ No grammar issues found.")
        return

    print(f"Found {len(matches)} issue(s):\\n")
    for match in matches:
        print(f"🔹 Issue: {match.message}")
        print(f"   Error in: '{text[match.offset:match.offset + match.errorLength]}'")
        if match.replacements:
            print(f"   Suggestion(s): {', '.join(match.replacements)}")
        print()
''',
            "python",
        ),
        (
            "Java",
            """
public class SimpleGrammarParser {

    private final String input;
    private int pos = 0;
    private List<String> arrayList = new ArrayList<>();

    public SimpleGrammarParser(String input) {
        this.input = input.replaceAll("\\s+", ""); // Remove spaces
        this.arrayList = List::new;
    }
""",
            "java",
        ),
        (
            "C++",
            """class MyClass {
public:
    int myField;
};

int main() {
    MyClass obj;
    obj.myField = 5;
}""",
            "cpp",
        ),
    ]

    # Test the first one (change index to test others)
    test_index = 1
    language_name, code, language = test_cases[test_index]

    print(f"=" * 60)
    print(f"Testing {language_name} Parser")
    print(f"=" * 60)
    print(f"\nSource code:\n{code}\n")
    print(f"=" * 60)

    cst, discard_reason = parser.parse(code, language)

    if cst is not None:
        print("Parsed CST:")
        cst.pretty()

        # print("\n\n=== SEMANTIC ANNOTATIONS ===\n")

        # annotated_nodes = []

        # for node in cst.traverse():
        #     if hasattr(node, 'semantic_label') and node.semantic_label is not None:
        #         annotated_nodes.append(node)

        # if annotated_nodes:
        #     print(f"Found {len(annotated_nodes)} annotated nodes:\n")
        #     for node in annotated_nodes:
        #         parent_type = node.parent.type if node.parent else "None"
        #         print(f"  [{node.semantic_label}] '{node.text}' (type: {node.type}, parent: {parent_type})")
        # else:
        #     print("No semantic annotations found.")

        # print("\n=== ALL IDENTIFIERS ===\n")
        # identifiers = [node for node in cst.traverse() if node.type == "identifier"]
        # print(f"Found {len(identifiers)} identifiers:\n")
        # for node in identifiers[:20]:  # Limit to first 20
        #     parent_type = node.parent.type if node.parent else "None"
        #     label = f"[{node.semantic_label}]" if hasattr(node, 'semantic_label') and node.semantic_label else "[unlabeled]"
        #     print(f"  {label} '{node.text}' (parent: {parent_type})")

    else:
        print(f"Invalid snippet, reason to discard: {discard_reason}")

    print(f"\n{'='*60}")
    print(f"To test other languages, change test_index in main():")
    print(f"  0 = Python, 1 = Java, 2 = C++")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
