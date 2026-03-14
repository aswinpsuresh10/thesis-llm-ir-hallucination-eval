# run_experiments.py
# ─────────────────────────────────────────────────────────────────────────────
# Master experiment runner.
# Instantiates all models and runs every pipeline stage end-to-end.
# Results are saved to the results/ directory automatically.
# ─────────────────────────────────────────────────────────────────────────────

import argparse

from src.models.gemini import GeminiModel
from src.models.gptj import GPTJ
from src.config import Models
from src.pipelines.parametric_recall import run_parametric_recall
from src.pipelines.veracity_classification import run_veracity_classification
from src.pipelines.claim_synthesis import run_claim_synthesis
from src.pipelines.llm_as_judge import run_llm_as_judge
from src.utils.logger import get_logger

logger = get_logger("run_experiments")


def parse_args():
    parser = argparse.ArgumentParser(description="Run thesis evaluation pipelines.")
    parser.add_argument("--dev",   action="store_true", help="Use small sample for dev/debug")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="Run only stage 1 (Parametric Recall) or 2 (Evidence-Based)")
    parser.add_argument("--model", type=str, choices=["gptj", "gemini", "all"], default="all",
                        help="Which model(s) to run")
    return parser.parse_args()


def main():
    args = parse_args()
    dev_samples = 100 if args.dev else None

    # Instantiate Models 
    models = {}
    if args.model in ("gptj", "all"):
        logger.info("Loading GPT-J …")
        models["gptj"] = GPTJ(load_in_8bit=True)

    if args.model in ("gemini", "all"):
        logger.info("Loading Gemini 2.0 Flash …")
        models["gemini_2_flash"] = GeminiModel(model_name=Models.GEMINI_2_FLASH)

        logger.info("Loading Gemini 2.5 Flash …")
        models["gemini_25_flash"] = GeminiModel(model_name=Models.GEMINI_25_FLASH)

    #  Stage 1: Parametric Recall 
    if args.stage in (None, 1):
        logger.info("=" * 60)
        logger.info("STAGE 1 — Parametric Recall (SimpleQA)")
        logger.info("=" * 60)

        for name, model in models.items():
            logger.info(f"Running Parametric Recall: {name}")
            run_parametric_recall(model, max_samples=dev_samples)

    #  Stage 2: Evidence-Based Verification 
    if args.stage in (None, 2):
        logger.info("=" * 60)
        logger.info("STAGE 2 — Evidence-Based Verification")
        logger.info("=" * 60)

        # GPT-J and Gemini 2.0 Flash are the generators (not 2.5 Flash)
        generator_keys = [k for k in models if k != "gemini_25_flash"]

        for dataset in ("fever", "scifact"):
            for name in generator_keys:
                model = models[name]

                #Veracity Classification
                logger.info(f"[2a] Veracity Classification | {name} | {dataset}")
                run_veracity_classification(model, dataset_name=dataset, max_samples=dev_samples)

                #Claim Synthesis
                logger.info(f"[2b] Claim Synthesis | {name} | {dataset}")
                run_claim_synthesis(model, dataset_name=dataset, max_samples=dev_samples)

                # LLM-as-a-Judge
                logger.info(f"[2c] LLM-as-a-Judge | {name} | {dataset}")
                auditor = models.get("gemini_25_flash")
                run_llm_as_judge(
                    generator=model,
                    dataset_name=dataset,
                    max_samples=dev_samples,
                    auditor=auditor,
                )

    logger.info("All experiments complete. Results saved to results/")


if __name__ == "__main__":
    main()