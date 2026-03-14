# SciFact dataset loader with label binarization.
#
# Original labels:   SUPPORTS | REFUTES | (no NEI in SciFact claims split)
# Binarized mapping:
#   SUPPORTS  → True   (Scientifically Validated)
#   REFUTES   → False  (Scientifically Contradicted)

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from src.config import RAW_DIR
from src.utils.logger import get_logger

logger = get_logger("dataset.scifact")

_CACHE_PATH = RAW_DIR / "scifact.parquet"

LABEL_MAP = {
    "SUPPORTS": True,
    "REFUTES":  False,
}


def load_scifact(
    split: str = "train",
    max_samples: int | None = None,
) -> pd.DataFrame:
    """
    Load SciFact from HuggingFace datasets (cached locally on first run).

    Returns a DataFrame with columns:
        - claim         : str   — the scientific claim text
        - label         : str   — original label (SUPPORTS / REFUTES)
        - binary_label  : bool  — True=SUPPORTS, False=REFUTES
        - evidence      : str   — concatenated evidence rationale sentences

    Args:
        split:          'train' or 'validation'
        max_samples:    Optionally cap row count for dev runs.
    """
    cache = _CACHE_PATH.with_stem(f"scifact_{split}")
    if cache.exists():
        logger.info(f"Loading SciFact ({split}) from cache: {cache}")
        df = pd.read_parquet(cache)
    else:
        logger.info(f"Downloading SciFact ({split}) from HuggingFace …")
        dataset = load_dataset("allenai/scifact", "claims", split=split)
        df = dataset.to_pandas()

        # Flatten evidence rationale sentences
        def _flatten_rationale(evidence_dict: dict) -> str:
            sentences = []
            for doc_id, entries in evidence_dict.items():
                for entry in entries:
                    sentences.extend(entry.get("sentences", []))
            return " ".join(str(s) for s in sentences) if sentences else ""

        df["evidence"] = df["evidence"].apply(_flatten_rationale)
        df = df.rename(columns={"claim": "claim", "cited_doc_ids": "doc_ids"})
        # SciFact uses 'gold_label' in the claims split
        if "gold_label" in df.columns:
            df = df.rename(columns={"gold_label": "label"})
        df["binary_label"] = df["label"].map(LABEL_MAP)
        df = df[["claim", "label", "binary_label", "evidence"]].dropna(
            subset=["label"]
        )
        df.to_parquet(cache, index=False)
        logger.info(f"SciFact ({split}) cached to {cache}")

    if max_samples:
        df = df.head(max_samples)

    logger.info(f"SciFact ({split}) loaded: {len(df)} samples")
    return df.reset_index(drop=True)