# Stage — LLM-as-a-Judge (Gemini 2.5 Flash as Auditor)
#
# Objective: Overcome the limitation of similarity-based metrics for logically
# complex labels (REFUTES, NOT ENOUGH INFO) where the generated claim is
# intentionally contradictory to the ground truth, making BLEU/BERTScore
# comparisons semantically invalid.
#
# Two-Stage Pipeline:
#   Generation  — Generator model (GPT-J or Gemini 2.0 Flash) produces a
#                 claim grounded in the provided evidence.
#   Verification — Auditor model (Gemini 2.5 Flash) performs blind review
#                  of the generated claim, returning True/False.
#
# Metric:  Generative Accuracy (Acc_gen) — fraction of claims judged True
# Models:  Generator: GPT-J, Gemini 2.0 Flash
#          Auditor:   Gemini 2.5 Flash (fixed across all experiments)
# Datasets: FEVER, SciFact (all labels — SUPPORTS, REFUTES, NEI)

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from src.config import PipelineConfig, RESULTS_DIR, Models
from src.datasets.fever import load_fever
from src.datasets.scifact import load_scifact
from src.evaluation.scoring import compute_generative_accuracy
from src.models.base import BaseLLM
from src.models.gemini import GeminiModel
from src.utils.io import save_metrics, save_results
from src.utils.logger import get_logger
from src.utils.prompts import ClaimSynthesisPrompts, LLMJudgePrompts

logger = get_logger("pipeline.llm_as_judge")


def run_llm_as_judge(
    generator: BaseLLM,
    dataset_name: str = "fever",
    split: str = "train",
    max_samples: int | None = None,
    batch_size: int | None = None,
    output_dir: Path | None = None,
    auditor: BaseLLM | None = None,
) -> dict:
    """
    Run the LLM-as-a-Judge pipeline.

    Workflow:
        1. Load full dataset (all labels including REFUTES / NEI)
        2. Generator produces a claim grounded in each evidence block
        3. Auditor reviews each generated claim — returns True / False
        4. Compute Generative Accuracy and Hallucination Rate
        5. Persist all records and metrics to disk

    Args:
        generator:    The model under evaluation (GPT-J or Gemini 2.0 Flash)
        dataset_name: 'fever' or 'scifact'
        split:        Dataset split
        max_samples:  Cap dataset size for dev runs
        batch_size:   Inference batch size for generator
        output_dir:   Results output directory
        auditor:      Auditor LLM (defaults to Gemini 2.5 Flash)

    Returns:
        Dict with generative_accuracy, hallucination_rate, and per-sample records.
    """
    bs = batch_size or PipelineConfig.SYNTHESIS_BATCH_SIZE
    gen_slug = generator.model_name.replace("/", "_")
    output_dir = output_dir or (
        RESULTS_DIR / "llm_as_judge" / dataset_name / gen_slug
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default auditor — Gemini 2.5 Flash
    if auditor is None:
        logger.info(f"Initialising default auditor: {Models.GEMINI_25_FLASH}")
        auditor = GeminiModel(model_name=Models.GEMINI_25_FLASH)

    # ── 1. Load Full Dataset (all labels) ─────────────────────────────────────
    if dataset_name == "fever":
        df = load_fever(split=split, include_nei=True, max_samples=max_samples)
    elif dataset_name == "scifact":
        df = load_scifact(split=split, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}.")

    # Drop rows with empty evidence (no generation possible without evidence)
    df = df[df["evidence"].str.strip().astype(bool)].reset_index(drop=True)
    logger.info(f"[LLM-Judge] Generator={generator} | {dataset_name} | {len(df)} samples")

    # ── 2. Generation Stage ───────────────────────────────────────────────────
    gen_prompts = [
        ClaimSynthesisPrompts.user(row["evidence"])
        for _, row in df.iterrows()
    ]
    generated_claims: list[str] = []
    for i in tqdm(range(0, len(gen_prompts), bs), desc=f"[LLM-Judge] Generation [{dataset_name}]"):
        batch = gen_prompts[i : i + bs]
        generated_claims.extend(
            generator.generate_batch(batch, max_new_tokens=PipelineConfig.SYNTHESIS_MAX_NEW_TOKENS)
        )

    # ── 3. Verification Stage (Auditor — Gemini 2.5 Flash) ───────────────────
    judge_prompts = [
        LLMJudgePrompts.user(claim) for claim in generated_claims
    ]
    judge_verdicts: list[str] = []
    for i in tqdm(range(0, len(judge_prompts), bs), desc=f"[LLM-Judge] Auditing [{dataset_name}]"):
        batch = judge_prompts[i : i + bs]
        judge_verdicts.extend(
            auditor.generate_batch(batch, max_new_tokens=PipelineConfig.JUDGE_MAX_NEW_TOKENS)
        )

    # ── 4. Compute Generative Accuracy ────────────────────────────────────────
    metrics = compute_generative_accuracy(judge_verdicts)

    # ── 5. Build Per-sample Records ───────────────────────────────────────────
    records = [
        {
            "evidence":         df.loc[i, "evidence"],
            "original_claim":   df.loc[i, "claim"],
            "original_label":   df.loc[i, "label"],
            "generated_claim":  generated_claims[i],
            "judge_verdict":    judge_verdicts[i],
            "is_faithful":      judge_verdicts[i].strip().lower() == "true",
        }
        for i in range(len(df))
    ]

    # ── 6. Persist ────────────────────────────────────────────────────────────
    save_results(records, output_dir / "predictions")
    save_metrics(metrics, output_dir / "metrics.json")

    logger.info(
        f"[LLM-Judge] {generator.model_name} | {dataset_name} | "
        f"GenAcc={metrics['generative_accuracy']:.3f} | "
        f"Hallucination={metrics['hallucination_rate']:.3f}"
    )
    return {"metrics": metrics, "records": records}