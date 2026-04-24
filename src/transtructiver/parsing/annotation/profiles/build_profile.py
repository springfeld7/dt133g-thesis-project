"""Utility script to build a lightweight builtin identifier profile.

This helper fetches pre-built documentation indices from DevDocs and
constructs a compact mapping from namespaces to commonly used type names
for a given language. The output is written as a JSON file under the
``parsing/annotation/profiles/<language>/`` directory. It is intended as
a convenience script for maintainer workflows rather than part of the
runtime pipeline.
"""

import os
import re
import requests
import json
from collections import defaultdict

_EXCLUDE_TYPES = [
    "Packages",
    "Versions",
    "Symbols",
    "Glossary",
    "Python/C API",
    "Tutorial",
]


# -----------------------------
# Helper: load DevDocs JSON
# -----------------------------
def load_devdocs(language: str, version: str | None = None) -> dict[str, list]:
    """Fetch the DevDocs index JSON for ``language``.

    DevDocs exposes a compact index usable to extract names and type
    categories. When ``version`` is provided it is requested using the
    DevDocs version selector syntax (``~<version>``).

    Args:
        language: Language key supported by DevDocs (e.g., "java", "python").
        version: Optional version string to select a specific docs build.

    Returns:
        Parsed JSON content as a dictionary with keys like "entries" and
        "types". Network errors will raise exceptions from ``requests``.
    """
    lang = "openjdk" if language == "java" else language
    v = f"~{version}" if version else ""
    return requests.get(f"https://devdocs.io/docs/{lang}{v}/index.json").json()


# -----------------------------
# Helper: build your profile format
# -----------------------------
def build_profile(entries: list, types: list, sep: str):
    """Build a namespace -> type-name profile from DevDocs entries.

    The function scans documentation entries and groups discovered type
    names under their predicted namespace (for example, ``java.lang`` ->
    ["String", "Object"]). The ``sep`` parameter controls how fully
    qualified names are split (``'.'`` for Java/Python, ``'::'`` for C++).

    Args:
        entries: List of documentation entry dicts from DevDocs index.
        types: Type/category mapping from DevDocs (unused here but kept for
               compatibility with DevDocs structure).
        sep: Namespace separator string used in qualified names.

    Returns:
        A dict mapping namespace -> sorted list of type names.
    """
    namespaces = defaultdict(set)

    for e in entries:
        full_name: str = e.get("name")
        if not language == "java" and (
            re.search(r"^(?![a-z0-9_])\w+(\s|-)+.+", full_name)
            or re.search(r"^(.*?>.*?)", full_name)
            or re.search(r"^[A-Z][a-z]+(?!\.)-?(?!\.)", full_name)
            or re.search(r"^.*?((?!\s+)\()(?!\)).*?", full_name)
            or re.search(r"^[^a-zA-Z(#|_)]+", full_name)
        ):
            continue

        t = e.get("type")
        if not t or t in _EXCLUDE_TYPES:
            continue

        words = re.split(r"\s+", full_name)
        name = words[0] if not t == "Keywords" else words[-1]

        if sep in name:
            ns, nm = name.rsplit(sep, 1)
            n = re.sub(r"[(\(\)):]+", "", nm)
            if ns != n:
                namespaces[ns].add(n)
                continue

        path = e.get("path")
        if not "/" in path:
            continue

        dr, n = path.rsplit("/", 1)
        if n.lower() == re.sub(r"[^\w]+", "", name).lower():
            p_split = dr.split("/")
            p = ".".join([s for s in p_split if not "." in s])
            namespaces[p].add(name)

    profile = {}

    for ns in sorted(namespaces.keys()):
        profile[ns] = sorted(namespaces[ns])

    return profile


# -----------------------------
# Fetch and build profile
# -----------------------------
language = "cpp"
version = None
separator = "::"

docs = load_devdocs(language, version)
entries = docs["entries"]
types = docs["types"]
profile = build_profile(entries, types, separator)

# -----------------------------
# Save to file
# -----------------------------
path = f"src/transtructiver/parsing/annotation/profiles/{language}/"
# Ensure the profile directory exists (create the language directory itself).
os.makedirs(path, exist_ok=True)

# Persist profile to a file. The filename includes the language and an
# optional version marker to distinguish profiles for different doc
# releases. This is a convenience step for maintainers.
with open(path + f"{language}{version if version else ' '}_builtin_profile.json", "w") as f:
    json.dump(profile, f, indent=2)

print(f"Generated {language}{version if version else ' '}_builtin_profile.json")
