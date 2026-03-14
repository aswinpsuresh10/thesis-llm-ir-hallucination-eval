# Central configuration for all models, datasets, and pipeline settings.
# Set your API keys in a .env file and they will be loaded here automatically.

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

#  Project Paths
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
RESULTS_DIR     = ROOT_DIR / "results"
LOGS_DIR        = ROOT_DIR / "logs"

for _dir in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

#  API Keys 
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

#  Model Identifiers 
class Models:
    GPTJ                = "EleutherAI/gpt-j-6b"
    GEMINI_2_FLASH      = "gemini-2.0-flash"
    GEMINI_25_FLASH     = "gemini-2.5-flash-preview-04-17"   # Auditor / Judge

# Dataset Names 
class Datasets:
    SIMPLEQA    = "simpleqa"
    FEVER       = "fever"
    SCIFACT     = "scifact"

#  Pipeline Settings 
class PipelineConfig:
    # Parametric Recall (SimpleQA)
    SIMPLEQA_SAMPLE_SIZE        = 4326     
    SIMPLEQA_MAX_NEW_TOKENS     = 50

    # Veracity Classification (FEVER / SciFact)
    VERACITY_MAX_NEW_TOKENS     = 10
    VERACITY_BATCH_SIZE         = 32

    # Claim Synthesis
    SYNTHESIS_MAX_NEW_TOKENS    = 128
    SYNTHESIS_BATCH_SIZE        = 16

    # LLM-as-a-Judge
    JUDGE_MAX_NEW_TOKENS        = 10
    JUDGE_MODEL                 = Models.GEMINI_25_FLASH

    # General
    RANDOM_SEED                 = 42
    LOG_LEVEL                   = "INFO"