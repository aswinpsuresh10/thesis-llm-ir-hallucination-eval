# SimpleQA dataset loader.
# 4,326 adversarially-challenging short-form QA pairs curated by OpenAI.
# Used exclusively for the Parametric Recall stage.

from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from pathlib import Path

from src.config import RAW_DIR
from src.utils.logger import get_logger

logger = get_logger("dataset.simpleqa")

_CACHE_PATH = RAW_DIR / "simpleqa.parquet"


def load_simpleqa(max_samples: int | None = None) -> pd.DataFrame:
    """
    Load SimpleQA from HuggingFace datasets (cached locally on first run).

    Returns a DataFrame with columns:
        - question      : str  — the question text
        - answer        : str  — the gold-standard short answer

    Args:
        max_samples: Optionally cap the number of rows (useful for dev runs).
    """
    if _CACHE_PATH.exists():
        logger.info(f"Loading SimpleQA from cache: {_CACHE_PATH}")
        df = pd.read_parquet(_CACHE_PATH)
    else:
        logger.info("Downloading SimpleQA from HuggingFace …")
        dataset = load_dataset("openai/simple-evals", "simpleqa", split="test")
        df = dataset.to_pandas()
        df = df.rename(columns={"problem": "question", "answer": "answer"})
        df = df[["question", "answer"]].dropna()
        df.to_parquet(_CACHE_PATH, index=False)
        logger.info(f"SimpleQA cached to {_CACHE_PATH}")

    if max_samples:
        df = df.head(max_samples)

    logger.info(f"SimpleQA loaded: {len(df)} samples")
    return df.reset_index(drop=True)