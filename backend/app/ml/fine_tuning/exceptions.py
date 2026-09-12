"""
Precious Edu LLM — Fine-Tuning Exception Hierarchy
"""


class FineTuningError(Exception):
    """Base exception for all fine-tuning pipeline failures."""
    pass


class DatasetValidationError(FineTuningError):
    """Raised when a conversational dataset record fails validation."""
    pass


class CheckpointCompatibilityError(FineTuningError):
    """Raised when a pretrained checkpoint architecture or tokenizer mismatches."""
    pass


class GenerationError(FineTuningError):
    """Raised when text generation or sampling fails."""
    pass
