"""analysis.py

Provides the SampleAnalyzer class for filtering and quantifying 
stylistic attributes of source code samples from the DroidCollection.
"""

from typing import Iterator, cast

from tree_sitter import Node as TSNode, Parser as TSParser
from tree_sitter_language_pack import SupportedLanguage, get_language
from transtructiver.parsing.parser import Parser


class SampleAnalyzer:
    """
    Orchestrates the analysis of source code snippets with configurable filters.

    This class handles gatekeeping logic and structural quantification. By
    injecting target criteria at initialization, the analyzer remains
    flexible for different experimental scopes.

    Attributes:
        target_languages (set): Set of languages (lowercase) to include.
        target_labels (set): Set of authorship labels (uppercase) to include.
        parser (Parser): An instance of TranStructIVer's Parser for future CST-based metrics.
    """

    def __init__(self, target_languages: set = None, target_labels: set = None):
        """
        Initializes the analyzer. If no criteria are passed, it defaults
        to the standard thesis scope (Python/Java/Cpp + Human/Machine).

        Args:
            target_languages (set, optional): A set of programming languages to filter by.
            target_labels (set, optional): A set of authorship labels to filter by.
        """
        # If nothing is passed, we use  specific thesis targets
        self.target_languages = target_languages or {"java", "python", "cpp"}

        # We normalize labels to uppercase to match the DroidCollection schema
        labels = target_labels or {"MACHINE_GENERATED", "HUMAN_GENERATED"}
        self.target_labels = {iter_var.upper() for iter_var in labels}

        self._ts_parser = TSParser()
        self._parser = Parser()

    def get_valid_tree(self, code: str, lang: str, label: str) -> TSNode | None:
        """
        Validates the sample against both language, authorship, and parsed tree
        requirements. Language and label must be within the experimental scope,
        and the code must successfully parse without error nodes.

        Args:
            code (str): The source code content to be analyzed.
            lang (str): The programming language identifier from the dataset.
            label (str): The authorship label (e.g., 'HUMAN_GENERATED').

        Returns:
            Node | None: The parsed tree if the sample is valid, otherwise None.
        """
        if not lang or not label or not (code and code.strip()):
            return None

        norm_lang = self._normalize_lang(lang)
        norm_label = str(label).strip().upper()
        if norm_lang not in self.target_languages or norm_label not in self.target_labels:
            return None

        self._ts_parser.language = get_language(cast(SupportedLanguage, norm_lang.lower()))
        tree = self._ts_parser.parse(bytes(code, "utf8")).root_node
        if not tree:
            return None

        for node in self._traverse(tree):
            if node.type == "ERROR" or self._parser.should_discard(tree, code):
                return None

        return tree

    def calculate_metrics(self, code: str, tree: TSNode) -> dict:
        """
        Calculates stylistic and structural metrics for a validated sample.

        Includes:
        - Character Count
        - Lines of Code (LOC)
        - Whitespace Ratio
        - For Loop Density
        - Comment Density
        - Identifier Density

        Args:
            code (str): The raw source code string to be analyzed.
            language (str): The validated language of the sample.
            label (str): The validated authorship label of the sample.
            tree (Node): The parsed CST of the code for structural metrics.

        Returns:
            dict: A collection of metrics including LOC and stylistic densities.
        """
        length = len(code)
        lines = code.splitlines()
        logical_lines = [line for line in lines if line.strip()]
        lloc_count = len(logical_lines)
        total_loc = len(lines)

        for_loops = 0
        identifiers = 0
        comments = 0

        for node in self._traverse(tree):
            if node.type == "for_statement":
                for_loops += 1
            elif node.type == "identifier":
                identifiers += 1
            elif self._is_comment(node):
                comments += node.end_point[0] - node.start_point[0] + 1

        # Count whitespace in gaps between nodes
        whitespace = self._count_whitespace_gaps(code, tree)

        # Helper to avoid DivisionByZero errors
        safe_lloc = lloc_count if lloc_count > 0 else 1
        safe_length = length if length > 0 else 1
        safe_loc = total_loc if total_loc > 0 else 1

        # Calculate Densities and Ratios
        metrics = {
            "char_count": length,
            "loc": total_loc,
            "lloc": lloc_count,
            "for_loop_density": for_loops / safe_lloc,
            "identifier_density": identifiers / safe_lloc,
            "comment_density": comments / safe_loc,
            "whitespace_ratio": whitespace / safe_length,
        }
        return metrics

    def _normalize_lang(self, lang: str) -> str:
        """
        Maps DroidCollection language strings to TranStructIVer standards.

        Example: 'C++' -> 'cpp', 'Java' -> 'java'
        """
        if not lang:
            return ""
        # Perform at least 1 insertion: replace C++ specifically
        return str(lang).strip().lower().replace("c++", "cpp")

    def _is_comment(self, node: TSNode):
        """Check if current not is a comment node."""
        return (
            node.type == "string"
            and node.parent
            and node.parent.type in {"module", "block"}
            and any(
                child.type in {"string_start", "string_end"} and child.text in ('"""', "'''")
                for child in node.children
            )
        ) or ("comment" in node.type)

    def _count_whitespace_gaps(self, code: str, tree: TSNode) -> int:
        """
        Count whitespace in gaps between tree nodes.

        Traverses the tree and tallies non-newline whitespace (with tab expansion)
        in byte ranges not occupied by named nodes.

        Args:
            code (str): The source code string.
            tree (TSNode): The parsed tree.

        Returns:
            int: Total weighted whitespace count (tabs = 4 spaces, excluding newlines).
        """
        code_bytes = code.encode("utf8")

        # Collect all byte ranges occupied by named nodes
        node_ranges = []
        for node in self._traverse(tree):
            # Skip the root node (it spans the entire file)
            if node.parent is None:
                continue
            if node.is_named:
                node_ranges.append((node.start_byte, node.end_byte))

        # Sort and merge overlapping ranges
        node_ranges.sort()
        merged_ranges = []
        for start, end in node_ranges:
            if merged_ranges and start <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
            else:
                merged_ranges.append((start, end))

        # Count whitespace in gaps
        whitespace = 0
        current_byte = 0
        for start, end in merged_ranges:
            if start > current_byte:
                gap_bytes = code_bytes[current_byte:start]
                gap_text = gap_bytes.decode("utf8")
                whitespace += self._tally_whitespace(gap_text)
            current_byte = end

        # Count trailing gap
        if current_byte < len(code_bytes):
            gap_bytes = code_bytes[current_byte:]
            gap_text = gap_bytes.decode("utf8")
            whitespace += self._tally_whitespace(gap_text)

        return whitespace

    def _tally_whitespace(self, text: str) -> int:
        """
        Tally whitespace characters (spaces, tabs) only, with tab expansion, excluding newlines.

        tabs are expanded to 4 spaces, spaces count as 1, all other chars are ignored.

        Args:
            text (str): Text to tally.

        Returns:
            int: Weighted whitespace count (only spaces and tabs).
        """
        count = 0
        for char in text:
            if char == " ":
                count += 1
            elif char == "\t":
                count += 4
        return count

    def _traverse(self, node: TSNode) -> Iterator[TSNode]:
        """
        Yield all nodes in the tree using preorder traversal.

        Yields:
            Node: Each node in the tree in preorder sequence.
        """
        yield node
        for child in node.children:
            yield from self._traverse(child)
