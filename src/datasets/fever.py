# FEVER dataset loader with label binarization.
#
# Original labels:   SUPPORTS | REFUTES | NOT ENOUGH INFO
# Binarized mapping:
#   SUPPORTS  → True   (Factually Accurate)
#   REFUTES   → False  (Factually Inaccurate)
#   NOT ENOUGH INFO → excluded from veracity classification
#                     but retained for LLM-as-a-Judge pipeline

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from src.config import RAW_DIR
from src.utils.logger import get_logger

logger = get_logger("dataset.fever")

_CACHE_PATH = RAW_DIR / "fever.parquet"

# Label mappings
LABEL_MAP = {
    "SUPPORTS": True,
    "REFUTES":  False,
}


def load_fever(
    split: str = "train",
    include_nei: bool = False,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """
    Load FEVER from HuggingFace datasets (cached locally on first run).

    Returns a DataFrame with columns:
        - claim         : str   — the claim text
        - label         : str   — original label (SUPPORTS / REFUTES / NEI)
        - binary_label  : bool  — True=SUPPORTS, False=REFUTES (NaN for NEI)
        - evidence      : str   — concatenated evidence sentences

    Args:
        split:          Dataset split: 'train', 'labelled_dev', or 'paper_test'
        include_nei:    If False (default), drops NOT ENOUGH INFO rows.
        max_samples:    Optionally cap row count for dev runs.
    """
    cache = _CACHE_PATH.with_stem(f"fever_{split}")
    if cache.exists():
        logger.info(f"Loading FEVER ({split}) from cache: {cache}")
        df = pd.read_parquet(cache)
    else:
        logger.info(f"Downloading FEVER ({split}) from HuggingFace …")
        dataset = load_dataset("fever", "v1.0", split=split, trust_remote_code=True)
        df = dataset.to_pandas()

        # Flatten evidence: concatenate all evidence sentences into one string
        def _flatten_evidence(evidence_sets) -> str:
            sentences = []
            for es in evidence_sets:
                for _, _, _, sent in es:
                    if sent:
                        sentences.append(sent)
            return " ".join(sentences) if sentences else ""

        df["evidence"] = df["evidence"].apply(_flatten_evidence)
        df = df.rename(columns={"claim": "claim", "label": "label"})
        df["binary_label"] = df["label"].map(LABEL_MAP)
        df = df[["claim", "label", "binary_label", "evidence"]]
        df.to_parquet(cache, index=False)
        logger.info(f"FEVER ({split}) cached to {cache}")

    if not include_nei:
        df = df[df["label"] != "NOT ENOUGH INFO"].copy()
        logger.info(f"Dropped NEI rows → {len(df)} rows remaining")

    if max_samples:
        df = df.head(max_samples)

    logger.info(f"FEVER ({split}) loaded: {len(df)} samples")
    return df.reset_index(drop=True)