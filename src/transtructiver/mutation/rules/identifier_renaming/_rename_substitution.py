"""Substitution helpers for identifier renaming.

This module chooses replacement identifier texts using a precomputed
frequency map keyed by semantic label and optional context type. The
selection logic prefers context-aware candidates but falls back to the
generic ("none") bucket when needed. The resulting candidate is then
formatted according to language naming conventions using
``format_identifier``.
"""

from operator import indexOf
import random

from transtructiver.node import Node
from transtructiver.mutation.rules.utils.formatter import split_words, format_identifier


# Canonical suffix map that collapses equivalent type names to shared tokens.
_SUFFIXES = {
"list": ["items", "elements", "arr", "array", "buffer", "sequence", "values", "collection", "vector", "series", "stream", "entries", "cluster", "bundle", "lineup", "store"],
"tuple": ["n_tuple", "tup", "coordinate", "immutable_list", "struct", "coord", "point", "record", "row", "data"],
"map": ["dictionary", "lookup_table", "kv_pairs", "object_map", "mapping", "lookup", "registry", "dict", "hash", "cache", "kv", "table"],
"set": ["unique_collection", "bag", "membership_list", "distinct_vals", "distinct_values", "unique_items", "seen", "visited", "distinct", "pool"],
"str": ["text", "message", "char_array", "chars", "msg", "lbl", "label", "raw", "content", "name"],
"num": ["value", "digit", "scalar", "count", "idx", "constant", "total", "index", "offset", "amt", "size", "nmb"],
"flag": ["toggle", "switch", "boolean", "bit", "indicator", "signal", "trigger", "check", "hint"],
"func": ["handler", "callback", "cb", "fn", "method", "routine", "subroutine", "procedure", "hook", "callback", "logic", "task"],
"cls": ["blueprint", "template", "type", "object", "model", "entity", "schema", "base", "impl", "wrapper", "component"],
"attr": ["property", "prop", "field", "meta", "member", "state", "val", "key", "info", "metadata"],
"var": ["reference", "pointer", "ref", "ptr", "identifier", "id", "bucket", "handle", "obj", "item", "temp", "res", "result", "entry"],
"param": ["input", "cfg", "option", "opt", "arg", "param", "requirement", "req", "placeholder"],
"arg": ["val", "passed_val", "data", "operand", "payload", "passed"],
}


_PREPOSITIONS = [
    "aboard",
    "about",
    "above",
    "absent",
    "across",
    "after",
    "against",
    "along",
    "alongside",
    "amid",
    "amidst",
    "among",
    "amongst",
    "around",
    "as",
    "astride",
    "at",
    "atop",
    "before",
    "afore",
    "behind",
    "below",
    "beneath",
    "beside",
    "besides",
    "between",
    "beyond",
    "by",
    "circa",
    "despite",
    "down",
    "during",
    "except",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "less",
    "like",
    "minus",
    "near",
    "nearer",
    "nearest",
    "of",
    "off",
    "on",
    "onto",
    "opposite",
    "outside",
    "over",
    "past",
    "per",
    "save",
    "since",
    "through",
    "throughout",
    "to",
    "toward",
    "towards",
    "under",
    "underneath",
    "until",
    "up",
    "upon",
    "upside",
    "versus",
    "via",
    "with",
    "within",
    "without",
]


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
    _rng = random.Random(42)

    if not node.text:
        return ""

    old_text = node.text
    words = split_words(old_text)
    rearranged = []
    last = None
    pos = 0

    for word in words:
        w = word.lower()

        # Keep context_type suffix fixed in last position.
        if w in _SUFFIXES and indexOf(words, word) == len(words) - 1:
            last = w
            continue

        # Ensure preposition is set with related words by moving insertion behind it.
        if w in _PREPOSITIONS:
            rearranged.insert(0, w)
            pos = 1
            continue

        # Rearrange words by inserting them at the start of the list.
        rearranged.insert(pos, w)
    
    if last:
        # Replace context_type suffix.
        subs = _SUFFIXES[last]
        rearranged.append(subs[_rng.randint(0, len(subs) - 1)])
        print(f"rearranged: {rearranged}")

    # Should always be false. If not, _rename_appendage failed.
    if all(word == old_text for word in rearranged):
        return old_text
    
    new_text = "_".join(rearranged)

    return format_identifier(node, new_text, language)
