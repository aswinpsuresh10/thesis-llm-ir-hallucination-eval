# Abstract base class that all model wrappers must implement.
# Ensures a uniform interface across GPT-J, Gemini 2.0 Flash, and
# Gemini 2.5 Flash regardless of their underlying API differences.

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Minimal interface contract for every LLM used in the thesis pipeline.
    Each concrete subclass handles its own authentication and batching logic.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate a single response for the given prompt string.

        Args:
            prompt:         The full input text (system + user merged or separate).
            max_new_tokens: Hard upper-bound on output length.

        Returns:
            Raw string response from the model.
        """
        ...

    @abstractmethod
    def generate_batch(
        self, prompts: list[str], max_new_tokens: int = 50
    ) -> list[str]:
        """
        Generate responses for a batch of prompts.
        Implementations should handle rate-limiting / chunking internally.

        Returns:
            List of responses in the same order as the input prompts.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}')"