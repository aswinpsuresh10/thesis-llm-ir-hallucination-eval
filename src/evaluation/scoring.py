# Scoring functions for all three pipeline stages.
#
# Stage 1 — SimpleQA multi-dimensional scoring (Correct / Incorrect / Not Attempted)
# Stage 2a — Strict Match veracity accuracy
# Stage 2c — Generative Accuracy (LLM-as-a-Judge)

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── SimpleQA Taxonomy ─────────────────────────────────────────────────────────

@dataclass
class SimpleQAResult:
    """
    Multi-dimensional reliability picture for one model on SimpleQA.
    Echoes the SimpleQA framework taxonomy to distinguish ignorance from error.
    """
    n_total:        int = 0
    n_correct:      int = 0
    n_incorrect:    int = 0
    n_not_attempted: int = 0

    # Detailed records for per-item analysis
    records: list[dict] = field(default_factory=list)

    @property
    def accuracy_overall(self) -> float:
        """Acc_overall = N_correct / N_total"""
        return self.n_correct / self.n_total if self.n_total else 0.0

    @property
    def hallucination_rate(self) -> float:
        """Fraction of questions answered incorrectly (not refused)."""
        return self.n_incorrect / self.n_total if self.n_total else 0.0

    @property
    def abstention_rate(self) -> float:
        """Fraction of questions the model refused or left empty."""
        return self.n_not_attempted / self.n_total if self.n_total else 0.0

    def to_dict(self) -> dict:
        return {
            "n_total":           self.n_total,
            "n_correct":         self.n_correct,
            "n_incorrect":       self.n_incorrect,
            "n_not_attempted":   self.n_not_attempted,
            "accuracy_overall":  round(self.accuracy_overall, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "abstention_rate":   round(self.abstention_rate, 4),
        }


# Refusal detection — patterns that indicate "I don't know"
_REFUSAL_PATTERNS = re.compile(
    r"\b(i don'?t know|i am not sure|i cannot|i can'?t|"
    r"unknown|not available|no answer|n/?a)\b",
    re.IGNORECASE,
)


def classify_simpleqa_response(
    response: str, gold_answer: str
) -> str:
    """
    Classify a single SimpleQA response as 'correct', 'incorrect', or
    'not_attempted' using the SimpleQA three-way taxonomy.

    A response is:
    - 'not_attempted' if it is empty or matches a known refusal pattern.
    - 'correct'       if the gold answer appears verbatim in the response.
    - 'incorrect'     otherwise (hallucination).
    """
    response = response.strip()

    if not response or _REFUSAL_PATTERNS.search(response):
        return "not_attempted"

    if gold_answer.lower() in response.lower():
        return "correct"

    return "incorrect"


# ── Veracity Classification Accuracy ─────────────────────────────────────────

def strict_match_veracity(prediction: str, gold_label: bool) -> int:
    """
    Strict Match protocol for veracity classification.

    Returns 1 if prediction strictly matches gold, else 0.
    Any response not equal to 'True' or 'False' is flagged as invalid (0).

    Args:
        prediction: Raw model output string.
        gold_label: Ground truth boolean.
    """
    pred = prediction.strip().lower()
    if pred == "true":
        pred_bool = True
    elif pred == "false":
        pred_bool = False
    else:
        return 0   # Invalid / non-compliant response

    return int(pred_bool == gold_label)


def compute_veracity_accuracy(predictions: list[str], gold_labels: list[bool]) -> dict:
    """
    Compute Acc_veracity = (1/n) * Σ I(Verdict_i == Gold_i) over all samples.
    Also reports the invalid response rate.
    """
    assert len(predictions) == len(gold_labels)

    scores, invalids = [], 0
    for pred, gold in zip(predictions, gold_labels):
        if pred.strip().lower() not in ("true", "false"):
            invalids += 1
            scores.append(0)
        else:
            scores.append(strict_match_veracity(pred, gold))

    n = len(scores)
    return {
        "n_total":          n,
        "n_correct":        sum(scores),
        "accuracy":         round(sum(scores) / n, 4) if n else 0.0,
        "invalid_rate":     round(invalids / n, 4) if n else 0.0,
    }


# ── Generative Accuracy (LLM-as-a-Judge) ─────────────────────────────────────

def compute_generative_accuracy(judge_verdicts: list[str]) -> dict:
    """
    Acc_gen = (1/n) * Σ I(Auditor(C_gen_i) == 'True')

    Args:
        judge_verdicts: List of raw auditor output strings ('True' / 'False').
    """
    n = len(judge_verdicts)
    n_true = sum(
        1 for v in judge_verdicts if v.strip().lower() == "true"
    )
    return {
        "n_total":              n,
        "n_judged_true":        n_true,
        "generative_accuracy":  round(n_true / n, 4) if n else 0.0,
        "hallucination_rate":   round(1 - n_true / n, 4) if n else 0.0,
    }