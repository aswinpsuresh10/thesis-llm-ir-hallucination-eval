# Centralised logger — logs to both console and a rotating file.

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import LOGS_DIR, PipelineConfig


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger writing to console + logs/<name>.log.
    Safe to call multiple times — handlers are not duplicated.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    level = getattr(logging, PipelineConfig.LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Rotating file handler (5 MB × 3 backups)
    log_file = LOGS_DIR / f"{name}.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger