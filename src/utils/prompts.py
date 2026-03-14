# Centralised prompt templates for all pipeline stages.
# All prompts are Zero-Shot and highly restrictive to isolate internal
# parametric recall and prevent Chain-of-Thought inflation.


class ParametricRecallPrompts:
    """
    Stage 1 — Parametric Recall (SimpleQA).
    Forces the model to rely solely on internal weights.
    No explanation, no full sentences — short-form factuality only.
    """

    SYSTEM = (
        "Answer this question using only the correct word or phrase. "
        "Do not explain. Do not use full sentences."
    )

    @staticmethod
    def user(question_text: str) -> str:
        return f"Question: {question_text}"


class VeracityClassificationPrompts:
    """
    Stage 2a — Direct Veracity Classification (FEVER / SciFact).
    Binary True/False task — no explanation, no hedging.
    Prevents Chain-of-Thought reasoning from inflating accuracy.
    """

    @staticmethod
    def user(claim_text: str) -> str:
        return (
            "Determine if this claim is True or False. "
            "Answer using only 'True' or 'False'. Do not explain.\n\n"
            f"Claim: {claim_text}"
        )


class ClaimSynthesisPrompts:
    """
    Stage 2b — Evidence-Driven Claim Synthesis (FEVER / SciFact SUPPORTS only).
    Model must generate a concise factual claim grounded in the provided evidence.
    No introductory phrasing to avoid polluting similarity metrics.
    """

    @staticmethod
    def user(evidence_text: str) -> str:
        return (
            "Write one concise factual claim supported by this evidence.\n\n"
            f"Evidence: {evidence_text}"
        )


class LLMJudgePrompts:
    """
    Stage 2c — LLM-as-a-Judge (Gemini 2.5 Flash as Auditor).
    Auditor performs blind review of generated claims.
    Constrained to yes/no to enable automated statistical analysis.
    """

    @staticmethod
    def user(generated_claim_text: str) -> str:
        return (
            "Determine if this claim is True or False. "
            "Answer using only 'True' or 'False'. Do not explain.\n\n"
            f"Claim: {generated_claim_text}"
        )