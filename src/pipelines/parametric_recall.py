# Stage — Parametric Recall (SimpleQA)
#
# Objective: Determine each model's hallucination floor by stress-testing
# internal parametric knowledge in a zero-shot short-form QA setting.
#
# Models evaluated: GPT-J (6B), Gemini 2.0 Flash, Gemini 2.5 Flash
# Dataset:          SimpleQA (4,326 adversarially-challenging questions)
# Scoring:          SimpleQA three-way taxonomy
#                   (Correct / Incorrect / Not Attempted)

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from src.config import PipelineConfig, RESULTS_DIR
from src.datasets.simpleqa import load_simpleqa
from src.evaluation.scoring import (
    SimpleQAResult,
    classify_simpleqa_response,
)
from src.models.base import BaseLLM
from src.utils.io import save_metrics, save_results
from src.utils.logger import get_logger
from src.utils.prompts import ParametricRecallPrompts

logger = get_logger("pipeline.parametric_recall")


def run_parametric_recall(
    model: BaseLLM,
    max_samples: int | None = None,
    batch_size: int = 32,
    output_dir: Path | None = None,
) -> SimpleQAResult:
    """
    Run the Parametric Recall evaluation pipeline on SimpleQA.

    Workflow:
        1. Load SimpleQA questions
        2. Build zero-shot prompts (system + question)
        3. Generate responses in batches
        4. Classify each response as Correct / Incorrect / Not Attempted
        5. Aggregate into a SimpleQAResult and persist to disk

    Args:
        model:        An instantiated BaseLLM (GPTJ, GeminiModel, …)
        max_samples:  Cap dataset size for dev runs (None = full 4,326)
        batch_size:   Number of prompts per inference batch
        output_dir:   Where to write CSV/JSON/metrics (default: results/parametric_recall/)

    Returns:
        SimpleQAResult with accuracy, hallucination rate, abstention rate.
    """
    output_dir = output_dir or RESULTS_DIR / "parametric_recall" / model.model_name.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    n = max_samples or PipelineConfig.SIMPLEQA_SAMPLE_SIZE
    df = load_simpleqa(max_samples=n)
    logger.info(f"[Parametric Recall] {model} | {len(df)} questions")

    # ── 2. Build Prompts ──────────────────────────────────────────────────────
    prompts = [
        f"{ParametricRecallPrompts.SYSTEM}\n\n{ParametricRecallPrompts.user(row['question'])}"
        for _, row in df.iterrows()
    ]

    # ── 3. Generate Responses in Batches ──────────────────────────────────────
    responses: list[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Parametric Recall"):
        batch = prompts[i : i + batch_size]
        responses.extend(
            model.generate_batch(batch, max_new_tokens=PipelineConfig.SIMPLEQA_MAX_NEW_TOKENS)
        )

    # ── 4. Classify Responses ─────────────────────────────────────────────────
    result = SimpleQAResult(n_total=len(df))
    records = []

    for idx, (_, row) in enumerate(df.iterrows()):
        label = classify_simpleqa_response(responses[idx], row["answer"])
        record = {
            "question":     row["question"],
            "gold_answer":  row["answer"],
            "response":     responses[idx],
            "label":        label,
        }
        records.append(record)

        if label == "correct":
            result.n_correct += 1
        elif label == "not_attempted":
            result.n_not_attempted += 1
        else:
            result.n_incorrect += 1

    result.records = records

    # ── 5. Persist ────────────────────────────────────────────────────────────
    save_results(records, output_dir / "predictions")
    save_metrics(result.to_dict(), output_dir / "metrics.json")

    logger.info(
        f"[Parametric Recall] {model.model_name} | "
        f"Acc={result.accuracy_overall:.3f} | "
        f"Hallucination={result.hallucination_rate:.3f} | "
        f"Abstention={result.abstention_rate:.3f}"
    )
    return result