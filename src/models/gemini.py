# Gemini wrapper supporting both Gemini 2.0 Flash (generator) and
# Gemini 2.5 Flash (auditor / LLM-as-a-Judge).
# Uses the google-genai SDK with automatic rate-limit retry.

from __future__ import annotations

import time

import google.generativeai as genai

from src.config import GOOGLE_API_KEY, Models
from src.models.base import BaseLLM
from src.utils.logger import get_logger

logger = get_logger("gemini")

# Retry configuration for rate-limit (429) errors
_MAX_RETRIES    = 5
_BACKOFF_BASE   = 2.0       # seconds; doubles on each retry


class GeminiModel(BaseLLM):
    """
    Wrapper for Google Gemini models via the google-genai SDK.

    Supports:
        - gemini-2.0-flash          (generator)
        - gemini-2.5-flash-preview  (auditor / judge)

    Pass model_name=Models.GEMINI_2_FLASH or Models.GEMINI_25_FLASH.
    """

    def __init__(self, model_name: str = Models.GEMINI_2_FLASH):
        super().__init__(model_name)
        if not GOOGLE_API_KEY:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )
        genai.configure(api_key=GOOGLE_API_KEY)
        self._client = genai.GenerativeModel(model_name)
        logger.info(f"Gemini model initialised: {model_name}")

    # ── Single Generation ─────────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Send a single prompt with exponential-backoff retry on 429."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_new_tokens,
                        temperature=0.0,        # Deterministic
                    ),
                )
                return response.text.strip()
            except Exception as exc:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    f"Gemini API error (attempt {attempt}/{_MAX_RETRIES}): "
                    f"{exc}. Retrying in {wait:.1f}s …"
                )
                time.sleep(wait)
        logger.error("Max retries exceeded for Gemini API call.")
        return ""

    # ── Batch Generation ──────────────────────────────────────────────────────

    def generate_batch(
        self, prompts: list[str], max_new_tokens: int = 50
    ) -> list[str]:
        """
        Sequential calls to the Gemini API.
        Gemini does not support true batch inference via the REST API,
        so requests are issued one by one with retry logic.
        """
        results = []
        for i, prompt in enumerate(prompts):
            if i % 100 == 0:
                logger.info(f"Gemini batch progress: {i}/{len(prompts)}")
            results.append(self.generate(prompt, max_new_tokens))
        return results