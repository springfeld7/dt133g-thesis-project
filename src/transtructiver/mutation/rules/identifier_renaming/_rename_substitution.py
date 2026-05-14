"""Substitution workflow for identifier renaming.

Provides semantic identifier renaming using masked language modeling and
cross-encoder similarity scoring. Leverages CodeBERT and VarCLR to generate
contextually-aware replacement names for identifiers in source code while
preserving semantic meaning.
"""

import random
import torch
import itertools
from typing import Optional
from difflib import SequenceMatcher
from functools import lru_cache
from transformers import AutoModelForMaskedLM, AutoTokenizer

from evaluation.varclr.models.encoders import Encoder
from ...mutation_context import MutationContext
from ....node import Node
from ..utils.formatter import format_identifier, split_words

context: MutationContext


def _get_resources(context: Optional[MutationContext]):
    """
    Load and cache ML models for identifier synonym generation.

    Initializes CodeBERT tokenizer/MLM model and VarCLR encoder on the first call.
    Results are cached globally to avoid redundant model loading.

    Returns:
        Tuple[AutoTokenizer, AutoModelForMaskedLM, Encoder, str]:
            Tokenizer, masked language model, VarCLR encoder, and device string.
    """
    if not context:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "microsoft/codebert-base-mlm"
    context.tokenizer = AutoTokenizer.from_pretrained(model_name)
    context.mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()

    context.varclr = Encoder.from_pretrained("varclr-codebert").to(device).eval()


@lru_cache(maxsize=512)
def _get_candidate_pool(
    word: str, context_code: str, tokenizer, mlm_model, top_n: int = 3
) -> list[str]:
    """
    Generate a semantically-similar synonym for an identifier using masked language modeling.

    Uses CodeBERT MLM to predict plausible replacements in context.

    Args:
        original (str): The original identifier name.
        context_code (str): Source code context containing the identifier.

    Returns:
        str: Pool of top-N semantic synonym.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mask only the specific sub-word in context
    masked_code = context_code.replace(word, tokenizer.mask_token)
    inputs = tokenizer(masked_code, return_tensors="pt", truncation=True, max_length=512).to(device)

    mask_idx = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
    if mask_idx.numel() == 0:
        return [word]

    with torch.no_grad():
        logits = mlm_model(**inputs).logits[0, mask_idx, :]
        top_tokens = torch.topk(logits, 20, dim=1).indices[0].tolist()

    del logits
    del inputs
    torch.cuda.empty_cache()

    candidates = []
    for t_id in top_tokens:
        cand = tokenizer.decode([t_id]).strip()
        # Quality filters
        if SequenceMatcher(a=word, b=cand).ratio() > 0.9:
            continue
        if cand.isalnum() and cand.lower() not in word.lower() and len(cand) > 1:
            candidates.append(cand)

        if len(candidates) >= top_n:
            break

    return candidates if candidates else [word]


def _build_substitute_name(node: Node, language: str, context: Optional[MutationContext]) -> str:
    """
    Generate a semantically-aware replacement name for an identifier node.

    Extracts the containing scope to provide context, splits the original identifier
    into constituent words, generates semantic synonyms for each word, and combines
    them back into a properly formatted identifier for the target language.

    Args:
        node (Node): The identifier node to rename.
        language (str): The programming language of the source code.

    Returns:
        str: A new identifier name formatted appropriately for the language,
            or an empty string if the node has no text.
    """
    if not node.text or not context:
        return ""

    if context.mlm_model is None or context.tokenizer is None or context.varclr is None:
        _get_resources(context)

    # Find nearest scoping ancestor
    context_node = next(
        (
            ancestor
            for ancestor in node.traverse_up()
            if (
                ancestor.semantic_label
                and "scope" in ancestor.semantic_label
                and len(ancestor.to_code()) > 50
            )
        ),
        node.parent or node,  # Fallback
    )

    code_string = context_node.to_code()
    original = node.text
    original_words = split_words(original)

    word_pool = []
    for word in original_words:
        assert context
        context_code = code_string[:100] if len(code_string) > 100 else code_string
        top_n = 2 if len(original_words) > 1 else 5
        word_pool.append(
            _get_candidate_pool(
                word, context_code, context.tokenizer, context.mlm_model, top_n=top_n
            )
        )

    # 3. Global Semantic Scoring via VarCLR
    with torch.no_grad():
        assert context.varclr
        new_words = []
        if len(original_words) > 1:
            # Generate all combinations (Cartesian Product)
            combos = itertools.product(*word_pool)
            new_words = [
                "_".join(combo)
                for combo in combos
                if context.varclr.score(combo[0], combo[1])[0] < 0.8
            ]
        else:
            new_words = word_pool[0]

        joined_original = "_".join(original_words)

        # Remove original from combinations if it snuck in
        new_words = [c for c in new_words if c != joined_original]

        if not new_words:
            return original

        # Score the FULL original name against all FULL combinations
        nested_scores = context.varclr.cross_score(joined_original, new_words)
        scores = torch.tensor(nested_scores[0], device="cpu")

        del nested_scores
        torch.cuda.empty_cache()

        # Pick from the top 3 overall performers for adversarial variety
        top_k = min(5, len(new_words))
        top_vals, top_indices = torch.topk(scores, top_k)

        # Keep only those within 0.1 of the best_score
        best_score = top_vals[0]
        competitive_mask = (best_score - top_vals) <= 0.1

        final_pool_indices = top_indices[competitive_mask].tolist()

        # Select one at random from the best
        final_idx = random.choice(final_pool_indices)
        final_name = new_words[final_idx]

    return format_identifier(node, final_name, language)
