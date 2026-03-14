# Stage — Direct Veracity Classification (FEVER / SciFact)
#
# Objective: Evaluate each model's internal fact-checking ability WITHOUT
# providing evidence context. Simulates a user asking a chatbot to verify
# a claim using the model's training knowledge alone.
#
# Task:       Binary True/False classification
# Constraint: NEI (Not Enough Info) labels excluded — binary task only
# Scoring:    Strict Match protocol (Acc_veracity)
# Models:     GPT-J (6B), Gemini 2.0 Flash
# Datasets:   FEVER, SciFact

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import PipelineConfig, RESULTS_DIR
from src.datasets.fever import load_fever
from src.datasets.scifact import load_scifact
from src.evaluation.scoring import compute_veracity_accuracy
from src.models.base import BaseLLM
from src.utils.io import save_metrics, save_results
from src.utils.logger import get_logger
from src.utils.prompts import VeracityClassificationPrompts

logger = get_logger("pipeline.veracity_classification")


def run_veracity_classification(
    model: BaseLLM,
    dataset_name: str = "fever",
    split: str = "train",
    max_samples: int | None = None,
    batch_size: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Run binary veracity classification on FEVER or SciFact.

    Workflow:
        1. Load dataset (NEI rows already filtered out)
        2. Build zero-shot prompts — no evidence provided
        3. Generate binary True/False responses in batches
        4. Compute Acc_veracity via Strict Match
        5. Persist predictions and metrics to disk

    Args:
        model:        An instantiated BaseLLM
        dataset_name: 'fever' or 'scifact'
        split:        Dataset split ('train', 'validation', etc.)
        max_samples:  Cap dataset size for dev runs
        batch_size:   Inference batch size (defaults to PipelineConfig value)
        output_dir:   Output directory (default: results/veracity_classification/)

    Returns:
        Dict with accuracy, invalid_rate, and per-sample records.
    """
    bs = batch_size or PipelineConfig.VERACITY_BATCH_SIZE
    model_slug = model.model_name.replace("/", "_")
    output_dir = output_dir or (
        RESULTS_DIR / "veracity_classification" / dataset_name / model_slug
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Dataset ───────────────────────────────────────────────────────
    if dataset_name == "fever":
        df = load_fever(split=split, include_nei=False, max_samples=max_samples)
    elif dataset_name == "scifact":
        df = load_scifact(split=split, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'fever' or 'scifact'.")

    logger.info(f"[Veracity] {model} | {dataset_name} | {len(df)} claims")

    # ── 2. Build Prompts (no evidence — parametric only) ──────────────────────
    prompts = [
        VeracityClassificationPrompts.user(row["claim"])
        for _, row in df.iterrows()
    ]

    # ── 3. Generate Responses ─────────────────────────────────────────────────
    responses: list[str] = []
    for i in tqdm(range(0, len(prompts), bs), desc=f"Veracity [{dataset_name}]"):
        batch = prompts[i : i + bs]
        responses.extend(
            model.generate_batch(batch, max_new_tokens=PipelineConfig.VERACITY_MAX_NEW_TOKENS)
        )

    # ── 4. Score ──────────────────────────────────────────────────────────────
    gold_labels = df["binary_label"].tolist()
    metrics = compute_veracity_accuracy(responses, gold_labels)

    # ── 5. Build Per-sample Records ───────────────────────────────────────────
    records = [
        {
            "claim":        df.loc[i, "claim"],
            "gold_label":   df.loc[i, "label"],
            "binary_label": gold_labels[i],
            "response":     responses[i],
            "correct":      responses[i].strip().lower() == str(gold_labels[i]).lower(),
        }
        for i in range(len(df))
    ]

    # ── 6. Persist ────────────────────────────────────────────────────────────
    save_results(records, output_dir / "predictions")
    save_metrics(metrics, output_dir / "metrics.json")

    logger.info(
        f"[Veracity] {model.model_name} | {dataset_name} | "
        f"Acc={metrics['accuracy']:.3f} | Invalid={metrics['invalid_rate']:.3f}"
    )
    return {"metrics": metrics, "records": records}