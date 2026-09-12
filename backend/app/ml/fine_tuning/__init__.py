"""
Precious Edu LLM — Fine-Tuning Module

Provides fine-tuning configuration, conversational dataset formatting, validator,
assistant-only loss masking, trainer engine, evaluator, and generation utilities.
"""

from app.ml.fine_tuning.config import FineTuningConfig, get_finetuning_config
from app.ml.fine_tuning.exceptions import (
    FineTuningError,
    DatasetValidationError,
    CheckpointCompatibilityError,
    GenerationError,
)

__all__ = [
    "FineTuningConfig",
    "get_finetuning_config",
    "FineTuningError",
    "DatasetValidationError",
    "CheckpointCompatibilityError",
    "GenerationError",
]
