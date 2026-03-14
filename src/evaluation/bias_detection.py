# Bias detection using WEAT and SEAT.
#
# WEAT  — Word Embedding Association Test  (Caliskan et al., 2017)
#          Measures implicit bias in static word embeddings.
#
# SEAT  — Sentence Encoder Association Test (May et al., 2019)
#          Extends WEAT to contextual sentence-level representations.
#
# Both tests compute an effect size (Cohen's d) and a p-value via
# permutation testing to determine statistical significance.

from __future__ import annotations

import itertools

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.logger import get_logger

logger = get_logger("bias_detection")

# Default encoder for SEAT
_DEFAULT_ENCODER = "all-roberta-large-v1"


# ── Cosine Similarity ─────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ── Association Score ─────────────────────────────────────────────────────────

def _association(w: np.ndarray, A: list[np.ndarray], B: list[np.ndarray]) -> float:
    """
    s(w, A, B) = mean_a cos(w,a) - mean_b cos(w,b)
    """
    return np.mean([_cosine(w, a) for a in A]) - np.mean([_cosine(w, b) for b in B])


def _effect_size(
    X: list[np.ndarray],
    Y: list[np.ndarray],
    A: list[np.ndarray],
    B: list[np.ndarray],
) -> float:
    """
    WEAT / SEAT effect size (Cohen's d):
    d = (mean_X s - mean_Y s) / std_XuY s
    """
    sx = [_association(x, A, B) for x in X]
    sy = [_association(y, A, B) for y in Y]
    numerator   = np.mean(sx) - np.mean(sy)
    denominator = np.std(sx + sy)
    return float(numerator / (denominator + 1e-10))


def _permutation_pvalue(
    X: list[np.ndarray],
    Y: list[np.ndarray],
    A: list[np.ndarray],
    B: list[np.ndarray],
    n_permutations: int = 10_000,
    seed: int = 42,
) -> float:
    """
    One-sided p-value via permutation test.
    """
    rng = np.random.default_rng(seed)
    observed = sum(
        _association(x, A, B) - _association(y, A, B)
        for x, y in zip(X, Y)
    )
    XY = X + Y
    n  = len(X)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(XY))
        Xi = [XY[i] for i in perm[:n]]
        Yi = [XY[i] for i in perm[n:]]
        stat = sum(
            _association(x, A, B) - _association(y, A, B)
            for x, y in zip(Xi, Yi)
        )
        if stat > observed:
            count += 1
    return count / n_permutations


# ── WEAT ──────────────────────────────────────────────────────────────────────

def weat(
    target_X: list[str],
    target_Y: list[str],
    attribute_A: list[str],
    attribute_B: list[str],
    embeddings: dict[str, np.ndarray],
    n_permutations: int = 10_000,
) -> dict:
    """
    Word Embedding Association Test (WEAT).

    Args:
        target_X:     Target concept X word list (e.g. male names)
        target_Y:     Target concept Y word list (e.g. female names)
        attribute_A:  Attribute A word list (e.g. career words)
        attribute_B:  Attribute B word list (e.g. family words)
        embeddings:   Pre-computed word → vector mapping
        n_permutations: Number of permutations for p-value

    Returns:
        dict with 'effect_size' and 'p_value'
    """
    def _lookup(words):
        return [embeddings[w] for w in words if w in embeddings]

    X, Y, A, B = _lookup(target_X), _lookup(target_Y), _lookup(attribute_A), _lookup(attribute_B)
    d = _effect_size(X, Y, A, B)
    p = _permutation_pvalue(X, Y, A, B, n_permutations)
    logger.info(f"WEAT | effect_size={d:.4f}, p_value={p:.4f}")
    return {"effect_size": round(d, 4), "p_value": round(p, 4)}


# ── SEAT ──────────────────────────────────────────────────────────────────────

def seat(
    target_X: list[str],
    target_Y: list[str],
    attribute_A: list[str],
    attribute_B: list[str],
    encoder_name: str = _DEFAULT_ENCODER,
    n_permutations: int = 10_000,
) -> dict:
    """
    Sentence Encoder Association Test (SEAT).
    Encodes all inputs with a SentenceTransformer and applies WEAT logic.

    Args:
        target_X:     Target concept X sentences
        target_Y:     Target concept Y sentences
        attribute_A:  Attribute A sentences
        attribute_B:  Attribute B sentences
        encoder_name: SentenceTransformer model to use for encoding
        n_permutations: Permutations for p-value

    Returns:
        dict with 'effect_size' and 'p_value'
    """
    logger.info(f"Running SEAT with encoder: {encoder_name}")
    encoder = SentenceTransformer(encoder_name)

    all_sentences = target_X + target_Y + attribute_A + attribute_B
    all_embeddings = encoder.encode(all_sentences, convert_to_numpy=True, show_progress_bar=False)

    idx = 0
    def _encode(texts):
        nonlocal idx
        vecs = [all_embeddings[idx + i] for i in range(len(texts))]
        idx += len(texts)
        return vecs

    X, Y, A, B = _encode(target_X), _encode(target_Y), _encode(attribute_A), _encode(attribute_B)

    d = _effect_size(X, Y, A, B)
    p = _permutation_pvalue(X, Y, A, B, n_permutations)
    logger.info(f"SEAT | effect_size={d:.4f}, p_value={p:.4f}")
    return {"effect_size": round(d, 4), "p_value": round(p, 4)}