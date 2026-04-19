import random

from ....node import Node
from ..utils.formatter import format_identifier
from ..utils.identifier_frequency_map import load_identifier_frequency_map


def _build_substitute_name(node: Node, language: str) -> str:
    """Generate substitution text for a node.

    Args:
        node: Identifier node being renamed.
        language: Language key resolved from CST root.

    Returns:
        The formatted identifier text after substitution.
    """
    _rng = random.Random(42)

    if not node.text:
        return ""

    old_text = node.text
    label = node.semantic_label
    frequency_map: dict[str, dict[str, int]] = {}

    if label and language in ["python", "java", "cpp"]:
        frequency_map = load_identifier_frequency_map(
            "output/test-identifier-frequency-map.json", language, label
        )

    if not frequency_map:
        return old_text

    identifiers: list[str] = []

    if node.context_type:
        identifier_map = frequency_map.get(node.context_type, {})
        for i in identifier_map.keys():
            identifiers.append(i)

    if node.context_type != "none" and len(identifiers) < 20:
        for i in frequency_map.get("none", {}).keys():
            identifiers.append(i)

    random_idx: int = _rng.randint(0, len(identifiers) - 1)
    new_text: str = identifiers[random_idx]

    return format_identifier(node, new_text, language)


def main():
    n = Node((0, 1), (0, 5), "identifier", "original_var", None)
    n.semantic_label = "variable_name"
    n.context_type = "number"

    print(_build_substitute_name(n, "cpp"))


if __name__ == "__main__":
    main()
