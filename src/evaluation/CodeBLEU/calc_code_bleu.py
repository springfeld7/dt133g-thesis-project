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


def calc_code_bleu(original_code, mutated_code, language) -> dict[str, float]:
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

    # preprocess inputs
    # pre_references = [
    #     [x.strip() for x in open(file, "r", encoding="utf-8").readlines()] for file in args.refs
    # ]
    references = [[original_code.strip()]]
    # hypothesis = [x.strip() for x in open(args.hyp, "r", encoding="utf-8").readlines()]
    hypothesis = [mutated_code.strip()]

    # for i in range(len(pre_references)):
    #     assert len(hypothesis) == len(pre_references[i])

    # references = []

    # for i in range(len(hypothesis)):
    #     ref_for_instance = []
    #     for j in range(len(pre_references)):
    #         ref_for_instance.append(pre_references[j][i])
    #     references.append(ref_for_instance)
    # assert len(references) == len(pre_references) * len(hypothesis)

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in ref] for ref in references]

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
        "dataflow_match_score": float(dataflow_match_score),
    }
