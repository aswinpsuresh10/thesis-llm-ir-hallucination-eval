# Stage — Evidence-Driven Claim Synthesis (FEVER / SciFact — SUPPORTS only)
#
# Objective: Assess faithfulness — the model's ability to synthesize a factual
# claim exclusively from provided evidence without introducing extrinsic
# hallucinations (external knowledge not present in the evidence).
#
# Task:       Source-to-claim generation
# Filter:     SUPPORTS labels only (evidence logically warrants the claim)
# Metrics:    BLEU, ROUGE-L, BERTScore F1
# Models:     GPT-J (6B), Gemini 2.0 Flash
# Datasets:   FEVER, SciFact

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import PipelineConfig, RESULTS_DIR
from src.datasets.fever import load_fever
from src.datasets.scifact import load_scifact
from src.evaluation.metrics import compute_all_metrics
from src.models.base import BaseLLM
from src.utils.io import save_metrics, save_results
from src.utils.logger import get_logger
from src.utils.prompts import ClaimSynthesisPrompts

logger = get_logger("pipeline.claim_synthesis")


def run_claim_synthesis(
    model: BaseLLM,
    dataset_name: str = "fever",
    split: str = "train",
    max_samples: int | None = None,
    batch_size: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Run Evidence-Driven Claim Synthesis on FEVER or SciFact.

    Workflow:
        1. Load dataset and filter to SUPPORTS only
        2. Build grounded prompts (evidence provided, no chain-of-thought)
        3. Generate one concise claim per evidence block
        4. Compute BLEU, ROUGE-L, BERTScore against ground-truth claims
        5. Persist predictions and metrics to disk

    Args:
        model:        An instantiated BaseLLM
        dataset_name: 'fever' or 'scifact'
        split:        Dataset split
        max_samples:  Cap dataset size for dev runs
        batch_size:   Inference batch size
        output_dir:   Output directory

    Returns:
        Dict with per-metric scores and per-sample records.
    """
    bs = batch_size or PipelineConfig.SYNTHESIS_BATCH_SIZE
    model_slug = model.model_name.replace("/", "_")
    output_dir = output_dir or (
        RESULTS_DIR / "claim_synthesis" / dataset_name / model_slug
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and Filter to SUPPORTS Only ──────────────────────────────────
    if dataset_name == "fever":
        df = load_fever(split=split, include_nei=False, max_samples=None)
    elif dataset_name == "scifact":
        df = load_scifact(split=split, max_samples=None)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}.")

    df = df[df["label"] == "SUPPORTS"].reset_index(drop=True)

    if max_samples:
        df = df.head(max_samples)

    # Drop rows with empty evidence
    df = df[df["evidence"].str.strip().astype(bool)].reset_index(drop=True)
    logger.info(f"[ClaimSynthesis] {model} | {dataset_name} | {len(df)} SUPPORTS samples")

    # ── 2. Build Grounded Prompts ─────────────────────────────────────────────
    prompts = [
        ClaimSynthesisPrompts.user(row["evidence"])
        for _, row in df.iterrows()
    ]

    # ── 3. Generate Claims ────────────────────────────────────────────────────
    generated_claims: list[str] = []
    for i in tqdm(range(0, len(prompts), bs), desc=f"Claim Synthesis [{dataset_name}]"):
        batch = prompts[i : i + bs]
        generated_claims.extend(
            model.generate_batch(batch, max_new_tokens=PipelineConfig.SYNTHESIS_MAX_NEW_TOKENS)
        )

    # ── 4. Compute Similarity Metrics ─────────────────────────────────────────
    reference_claims = df["claim"].tolist()
    metrics = compute_all_metrics(generated_claims, reference_claims)

    # ── 5. Build Per-sample Records ───────────────────────────────────────────
    bleu_scores  = metrics["bleu"]["per_sample"]
    rouge_scores = metrics["rouge_l"]["per_sample"]
    bert_scores  = metrics["bert_score_f1"]["per_sample"]

    records = [
        {
            "evidence":         df.loc[i, "evidence"],
            "reference_claim":  reference_claims[i],
            "generated_claim":  generated_claims[i],
            "bleu":             bleu_scores[i],
            "rouge_l":          rouge_scores[i],
            "bert_score_f1":    bert_scores[i],
        }
        for i in range(len(df))
    ]

    # ── 6. Persist ────────────────────────────────────────────────────────────
    summary_metrics = {
        "bleu_mean":            metrics["bleu"]["mean"],
        "rouge_l_mean":         metrics["rouge_l"]["mean"],
        "bert_score_f1_mean":   metrics["bert_score_f1"]["mean"],
        "n_samples":            len(df),
    }
    save_results(records, output_dir / "predictions")
    save_metrics(summary_metrics, output_dir / "metrics.json")

    logger.info(
        f"[ClaimSynthesis] {model.model_name} | {dataset_name} | "
        f"BLEU={summary_metrics['bleu_mean']:.3f} | "
        f"ROUGE-L={summary_metrics['rouge_l_mean']:.3f} | "
        f"BERTScore={summary_metrics['bert_score_f1_mean']:.3f}"
    )
    return {"metrics": summary_metrics, "records": records}