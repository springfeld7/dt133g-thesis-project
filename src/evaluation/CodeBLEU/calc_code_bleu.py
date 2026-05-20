# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
# import argparse
from . import bleu
from . import weighted_ngram_match
from . import syntax_match
from . import dataflow_match

# parser = argparse.ArgumentParser()
# parser.add_argument("--refs", type=str, nargs="+", required=True, help="reference files")
# parser.add_argument("--hyp", type=str, required=True, help="hypothesis file")
# parser.add_argument(
#     "--lang",
#     type=str,
#     required=True,
#     choices=["java", "javascript", "c_sharp", "php", "go", "python", "cpp", "c", "ruby"],
#     help="programming language",
# )
# parser.add_argument(
#     "--params", type=str, default="0.25,0.25,0.25,0.25", help="alpha, beta and gamma"
# )

# args = parser.parse_args()


def calc_code_bleu(original_code: list[str], mutated_code: str, language: str) -> dict[str, float]:
    """
    Calculates the CodeBLEU metric score for a single prediction snippet 
    against a list of valid reference code snippets.

    Args:
        original_code (list[str]): A list of reference strings representing valid solutions.
        mutated_code (str): The mutated or generated code string to evaluate.
        language (str): The programming language of the snippets (e.g., 'python').

    Returns:
        dict[str, float]: A dictionary containing the composite codebleu score and 
            the individual sub-scores (ngram_match_score, weighted_ngram_match_score, 
            syntax_match_score, dataflow_match_score).
    """
    lang = language
    params = "0.25,0.25,0.25,0.25"
    alpha, beta, gamma, theta = [float(x) for x in params.split(",")]

    # Preprocess inputs as clean, monolithic text blocks
    hypothesis = [mutated_code.strip()]
    references = [[ref.strip() for ref in original_code]]

    # Clean whitespace tokenization for standard n-gram matching
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # Calculate weighted ngram match using keyword tables
    keywords = [
        x.strip()
        for x in open(
            "src/evaluation/CodeBLEU/keywords/" + lang + ".txt",
            "r",
            encoding="utf-8",
        ).readlines()
    ]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [
        [
            [reference_tokens, make_weights(reference_tokens, keywords)]
            for reference_tokens in reference
        ]
        for reference in tokenized_refs
    ]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
        tokenized_refs_with_weights, tokenized_hyps
    )

    # Calculate syntax tree and dataflow graph matches
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    code_bleu_score = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * syntax_match_score
        + theta * dataflow_match_score
    )

    return {
        "codebleu": float(code_bleu_score),
        "ngram_match_score": float(ngram_match_score),
        "weighted_ngram_match_score": float(weighted_ngram_match_score),
        "syntax_match_score": float(syntax_match_score),
        "dataflow_match_score": float(dataflow_match_score)
    }

def calc_dataset_average_codebleu(references: list[str], predictions: list[str], lang: str) -> dict[str, float]:
    """
    Computes CodeBLEU for each 1:1 pair individually and returns the arithmetic mean.

    Args:
        references (list[str]): List of original code snippets.
        predictions (list[str]): List of mutated code snippets (must align 1:1 with references).
        lang (str): Target language string ('python').

    Returns:
        dict[str, float]: The average of all sub-scores across the dataset.
    """
    assert len(references) == len(predictions), "Data arrays must be a strict 1:1 length match."
    
    total_scores = {
        "codebleu": 0.0,
        "ngram_match_score": 0.0,
        "weighted_ngram_match_score": 0.0,
        "syntax_match_score": 0.0,
        "dataflow_match_score": 0.0
    }
    
    successful_pairs = 0

    for ref_snippet, pred_snippet in zip(references, predictions):
        try:
            pair_score = calc_code_bleu(
                original_code=[ref_snippet],
                mutated_code=pred_snippet, 
                language=lang
            )

            # Extract scores from the print/return layout of your single function
            total_scores["codebleu"] += pair_score["codebleu"]
            total_scores["ngram_match_score"] += pair_score["ngram_match_score"]
            total_scores["weighted_ngram_match_score"] += pair_score["weighted_ngram_match_score"]
            total_scores["syntax_match_score"] += pair_score["syntax_match_score"]
            total_scores["dataflow_match_score"] += pair_score["dataflow_match_score"]
            
            successful_pairs += 1
            
        except Exception as e:
            # If a mutation creates completely invalid syntax that tree-sitter crashes on,
            # we log it and keep going so the whole notebook doesn't die.
            print(f"Skipping a pair due to processing error: {e}")
            continue

    if successful_pairs == 0:
        return {k: 0.0 for k in total_scores}

    # Calculate final arithmetic mean
    return {metric: total / successful_pairs for metric, total in total_scores.items()}
