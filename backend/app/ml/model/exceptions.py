"""
Precious Edu LLM — Transformer Model Exceptions

Custom exception classes for model architecture, configuration, loading, and generation.
"""


class ModelError(Exception):
    """Base exception for all model errors."""
    pass


class ModelConfigError(ModelError):
    """Raised when model configuration is invalid or missing required fields."""
    pass


class CheckpointError(ModelError):
    """Raised when model checkpoint saving or loading fails."""
    pass


class ContextLengthExceededError(ModelError):
    """Raised when input sequence length exceeds configured max sequence length."""
    pass


class TokenizerIncompatibilityError(ModelError):
    """Raised when model configuration does not match tokenizer vocabulary or special tokens."""
    pass
