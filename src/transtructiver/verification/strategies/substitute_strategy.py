"""substitute_strategy.py

Defines the SubstituteVerificationStrategy for validating structural swaps (e.g., For -> While).
"""

from typing import List, Optional
from ...mutation.mutation_manifest import ManifestEntry
from ...node import Node
from .verification_strategy import VerificationStrategy


class SubstituteVerificationStrategy(VerificationStrategy):
    """
    Validation logic for structural substitutions.

    This strategy ensures that a node has been correctly transformed from its original type
    to a new type as specified in the mutation manifest. It checks for the presence of the mutated node,
    validates the type transition, and confirms that the content matches the expected new value.
    """

    def verify(self, orig: Optional[Node], mut: Optional[Node], entry: ManifestEntry) -> List[str]:
        """
        Audits a substitution by checking type transitions and content consistency.

        This method validates that a node has transitioned correctly from its
        original state (orig) to its mutated state (mut) according to the
        metadata stored in the manifest.

        Args:
            orig (Optional[Node]): The node as it existed in the original CST.
            mut (Optional[Node]): The corresponding node in the mutated tree.
            entry (ManifestEntry): The registry entry containing 'old_type',
                'new_type', and 'new_val' metadata.

        Returns:
            List[str]: A list of error messages; empty if verification passes.
        """
        errors = []

        # Physical Existence Checks
        if orig is None:
            return [f"SUBSTITUTE Error: Original node missing for ID {entry.original_id}"]
        if mut is None:
            return [f"SUBSTITUTE Error: Mutated node missing for ID {entry.original_id}"]

        # Metadata Extraction
        expected_new_type = entry.metadata.get("new_type")
        expected_new_val = entry.metadata.get("new_val")
        recorded_old_type = entry.metadata.get("old_type")

        # Verify the Manifest vs. The Original Tree
        if orig.type != recorded_old_type:
            errors.append(
                f"Manifest Mismatch: Rule recorded 'old_type' as {recorded_old_type}, "
                f"but the actual original node was {orig.type}."
            )

        if mut.type != expected_new_type:
            errors.append(
                f"Type Mismatch: Expected mutated type '{expected_new_type}', "
                f"but found '{mut.type}'."
            )

        if mut.text != expected_new_val:
            errors.append(
                f"Content Mismatch at {entry.original_id}: "
                f"Expected '{expected_new_val}', but found '{mut.text}'."
            )

        # Check for valid type transitions
        if not self._is_valid_transformation(orig.type, mut.type):
            errors.append(
                f"Invalid Logic Transition: Replacing '{orig.type}' with "
                f"'{mut.type}' is not a valid substitution."
            )

        return errors

    def _is_valid_transformation(self, old_type: str, new_type: str) -> bool:
        """
        Determines if a type transition is a recognized isomorphic substitution.

        This acts as a whitelist to prevent accidental structural corruptions
        (e.g., substituting a loop with a variable declaration).

        Args:
            old_type (str): The node type prior to mutation.
            new_type (str): The node type after mutation.

        Returns:
            bool: True if the transition is logically permissible.
        """
        # Map of (source_type) -> (allowed_target_types)
        valid_pairs = {
            ("for_statement", "while_statement"),
            ("for", "while"),
            ("enhanced_for_statement", "while_statement"),
            ("in", "true"),
        }
        return (old_type, new_type) in valid_pairs
