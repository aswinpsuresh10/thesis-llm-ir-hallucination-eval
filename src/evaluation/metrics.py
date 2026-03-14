# Semantic similarity metrics for the Claim Synthesis stage.
# Three complementary metrics each capturing a different facet of faithfulness:
#
#   BLEU      — n-gram precision overlap
#   ROUGE-L   — longest common subsequence recall
#   BERTScore — contextual embedding similarity
#
# Each metric is applied to (generated_claim, ground_truth_claim) pairs.

from __future__ import annotations

import numpy as np
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from src.utils.logger import get_logger

logger = get_logger("metrics")

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_SMOOTH = SmoothingFunction().method1


# ── Per-sample Scores ─────────────────────────────────────────────────────────

def bleu_score(hypothesis: str, reference: str) -> float:
    """
    Sentence-level BLEU (1–4 gram) with add-1 smoothing.
    Returns a float in [0, 1].
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens  = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_SMOOTH)


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L F1 score between hypothesis and reference.
    Returns a float in [0, 1].
    """
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    scores = _ROUGE_SCORER.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


# ── Corpus-level BERTScore ────────────────────────────────────────────────────

def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    lang: str = "en",
    model_type: str = "roberta-large",
    batch_size: int = 64,
) -> list[float]:
    """
    Compute BERTScore F1 for a list of (hypothesis, reference) pairs.
    Runs on GPU if available.

    Returns:
        List of per-sample F1 scores in [0, 1].
    """
    logger.info(f"Computing BERTScore for {len(hypotheses)} pairs …")
    _, _, F1 = bert_score_fn(
        hypotheses,
        references,
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
    )
    return F1.tolist()


# ── Aggregate All Metrics ─────────────────────────────────────────────────────

def compute_all_metrics(
    hypotheses: list[str],
    references: list[str],
) -> dict:
    """
    Compute BLEU, ROUGE-L, and BERTScore for all (hypothesis, reference) pairs.

    Returns:
        Dict containing per-sample scores and corpus-level averages.
    """
    assert len(hypotheses) == len(references), "Lengths must match."

    bleu_scores   = [bleu_score(h, r) for h, r in zip(hypotheses, references)]
    rouge_scores  = [rouge_l_score(h, r) for h, r in zip(hypotheses, references)]
    bert_scores   = compute_bertscore(hypotheses, references)

    return {
        "bleu":         {"per_sample": bleu_scores,  "mean": round(float(np.mean(bleu_scores)), 4)},
        "rouge_l":      {"per_sample": rouge_scores, "mean": round(float(np.mean(rouge_scores)), 4)},
        "bert_score_f1":{"per_sample": bert_scores,  "mean": round(float(np.mean(bert_scores)), 4)},
    }