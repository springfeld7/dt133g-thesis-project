"""Substitution helpers for identifier renaming.

This module chooses replacement identifier texts using a precomputed
frequency map keyed by semantic label and optional context type. The
selection logic prefers context-aware candidates but falls back to the
generic ("none") bucket when needed. The resulting candidate is then
formatted according to language naming conventions using
``format_identifier``.
"""

import random
from difflib import SequenceMatcher

from ....node import Node
from ..utils.formatter import format_identifier
from ..utils.identifier_frequency_map import load_identifier_frequency_map


def _build_substitute_name(node: Node, language: str) -> str:
    """Generate a substitution text for an identifier node.

    Selection strategy:
      1. If a precompiled frequency map exists for the node's semantic
         label and language, attempt to pick a candidate from the map.
      2. Prefer candidates that match the identifier's ``context_type``
         (for example, the declared type of a variable). If too few
         context-specific candidates are available, augment the pool with
         generic candidates from the ``"none"`` bucket.
      3. Pick a candidate deterministically using a fixed PRNG seed so
         unit tests and builds are reproducible.

    Args:
        node: The identifier node being renamed. Useful fields are
            ``semantic_label`` (e.g. "variable_name") and ``context_type``
            (inferred type like "list").
        language: Language key (e.g. "python", "java", "cpp").

    Returns:
        The new identifier string formatted to match language conventions.
        If no frequency map is available, the original text is returned.
    """
    # Deterministic RNG for reproducible name selection in tests.
    _rng = random.Random(42)

    if not node.text:
        return ""

    old_text = node.text
    label = node.semantic_label
    frequency_map: dict[str, dict[str, int]] = {}

    # Load a small role-keyed frequency map produced by the offline build
    # step. The map structure is: { context_type -> { identifier -> count } }
    if label and language in ["python", "java", "cpp"]:
        frequency_map = load_identifier_frequency_map(
            "output/identifier-frequency-map.json", language, label
        )

    # No external map available -> keep original name.
    if not frequency_map:
        return old_text

    # Build candidate pool: prefer context-specific identifiers first.
    identifiers: list[str] = []

    # Similarity thresholds
    MIN_SIMILARITY = 0.3
    MAX_SIMILARITY = 0.9

    def is_similar(a: str, b: str) -> bool:
        ratio = SequenceMatcher(None, a, b).ratio()
        print(f"Ratio {ratio}")
        return MIN_SIMILARITY <= ratio <= MAX_SIMILARITY

    length_in_range = lambda x: len(x) in range(len(old_text) - 2, len(old_text) + 2)

    if node.context_type:
        identifier_map = frequency_map.get(node.context_type, {})
        for i in identifier_map.keys():
            if length_in_range(i):
                identifiers.append(i)

    # If context-specific pool is small, augment with generic identifiers.
    if node.context_type != "none" and len(identifiers) < 20:
        for i in frequency_map.get("none", {}).keys():
            if length_in_range(i) and len(identifiers) < 40:
                identifiers.append(i)

    if not identifiers:
        return old_text

    similar_identifiers = []

    for i in identifiers:
        if is_similar(old_text, i):
            similar_identifiers.append(i)

    if similar_identifiers:
        random_idx: int = _rng.randint(0, len(similar_identifiers) - 1)
        new_text: str = similar_identifiers[random_idx]
    else:
        random_idx: int = _rng.randint(0, len(identifiers) - 1)
        new_text: str = identifiers[random_idx]

    return format_identifier(node, new_text, language)
