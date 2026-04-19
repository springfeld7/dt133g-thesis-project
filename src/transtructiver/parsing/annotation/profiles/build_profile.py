import os
import re

from numpy import full
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
    lang = "openjdk" if language == "java" else language
    v = f"~{version}" if version else ""
    return requests.get(f"https://devdocs.io/docs/{lang}{v}/index.json").json()


# -----------------------------
# Helper: build your profile format
# -----------------------------
def build_profile(entries: list, types: list, sep: str):
    """
    sep = '.' for Java/Python
    sep = '::' for C++
    Output format:
    {
        "java": ["lang", "util", ...],
        "lang": ["String", "Object", ...],
        "util": ["List", "ArrayList", ...]
    }
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
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path + f"{language}{version if version else " "}_builtin_profile.json", "w") as f:
    json.dump(profile, f, indent=2)

print(f"Generated {language}{version if version else " "}_builtin_profile.json")
