"""analysis.py

Provides the SampleAnalyzer class for filtering and quantifying 
stylistic attributes of source code samples from the DroidCollection.
"""

from transtructiver.node import Node
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
        
        self._parser = Parser()

    def get_valid_tree(self, code: str, lang: str, label: str) -> Node | None:
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

        tree, _ = self._parser.parse(code, norm_lang)
        if not tree:
            return None

        for node in tree.traverse():
            if node.type == "ERROR":
                return None

        return tree

    def calculate_metrics(self, code: str, language: str, label: str, tree: Node) -> dict:
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
        whitespace = 0

        for node in tree.traverse():
            if node.type == "for_statement":
                for_loops += 1
            elif node.type == "identifier":
                identifiers += 1
            elif bool(node.semantic_label) and "comment" in node.semantic_label:
                comments += node.end_point[0] - node.start_point[0] + 1
            elif node.type == "whitespace":
                content = node.text if node.text else ""
                weighted_content = content.replace('\t', '    ')
                whitespace += len(weighted_content)

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
            "whitespace_ratio": whitespace / safe_length
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
