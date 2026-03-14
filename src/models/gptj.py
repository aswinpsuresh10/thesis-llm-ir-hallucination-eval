# GPT-J (6B) wrapper using HuggingFace Transformers.
# Loaded in 8-bit quantization (bitsandbytes) to fit on a single GPU.

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import HUGGINGFACE_TOKEN, Models, PipelineConfig
from src.models.base import BaseLLM
from src.utils.logger import get_logger

logger = get_logger("gptj")


class GPTJ(BaseLLM):
    """
    Wrapper around EleutherAI/gpt-j-6b loaded via HuggingFace Transformers.

    The model is loaded once at instantiation and kept in memory.
    Use load_in_8bit=True to reduce VRAM to ~8 GB.
    """

    def __init__(self, load_in_8bit: bool = True):
        super().__init__(Models.GPTJ)
        logger.info(f"Loading {self.model_name} (8-bit={load_in_8bit}) …")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=HUGGINGFACE_TOKEN or None,
        )
        # GPT-J has no official pad token — reuse eos
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit=load_in_8bit,
            device_map="auto",
            token=HUGGINGFACE_TOKEN or None,
        )
        self.model.eval()
        logger.info("GPT-J loaded successfully.")

    # ── Single Generation ─────────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return self.generate_batch([prompt], max_new_tokens)[0]

    # ── Batch Generation ──────────────────────────────────────────────────────

    def generate_batch(
        self, prompts: list[str], max_new_tokens: int = 50
    ) -> list[str]:
        """
        Run inference on a list of prompts.
        Pads to the longest sequence in the batch.
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # Greedy decoding for reproducibility
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude the prompt)
        input_len = inputs["input_ids"].shape[1]
        responses = self.tokenizer.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )
        return [r.strip() for r in responses]