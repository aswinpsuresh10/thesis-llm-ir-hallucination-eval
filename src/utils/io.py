# Helpers for saving and loading pipeline results (CSV / JSON).

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("io")


def save_results(data: list[dict], path: Path) -> None:
    """Persist a list of result dicts as CSV + JSON side-by-side."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(path.with_suffix(".csv"), index=False)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(data)} records → {path}")


def load_results(path: Path) -> list[dict]:
    """Load results from CSV (preferred) or JSON."""
    path = Path(path)
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv")).to_dict(orient="records")
    if path.with_suffix(".json").exists():
        with open(path.with_suffix(".json")) as f:
            return json.load(f)
    raise FileNotFoundError(f"No results found at {path}")


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Persist a metrics dictionary as a pretty-printed JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {path}")