"""
Precious Edu LLM — Pretraining Exceptions

Custom exception classes for dataset loading, training loop, numerical stability,
device placement, and resume operations.
"""


class TrainingError(Exception):
    """Base exception for all pretraining errors."""
    pass


class DataLoadingError(TrainingError):
    """Raised when loading or validating tokenized dataset files fails."""
    pass


class NumericalInstabilityError(TrainingError):
    """Raised when NaN or Inf values are detected in loss, gradients, or parameters."""
    pass


class DeviceMismatchError(TrainingError):
    """Raised when tensors or modules reside on incompatible devices."""
    pass


class TrainingInterruptedError(TrainingError):
    """Raised when training is interrupted by user or emergency stop."""
    pass
