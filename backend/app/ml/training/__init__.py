"""
Precious Edu LLM — Training Package

Exports training configuration, exceptions, dataset loader, optimizer, scheduler,
checkpointing, evaluator, metrics logger, and pretraining engine.
"""

from app.ml.training.config import TrainingConfig
from app.ml.training.exceptions import (
    TrainingError,
    DataLoadingError,
    NumericalInstabilityError,
    DeviceMismatchError,
    TrainingInterruptedError,
)
from app.ml.training.reproducibility import set_seed
from app.ml.training.data_pipeline import TokenizedDataset, create_dataloader
from app.ml.training.optimizer import configure_optimizer
from app.ml.training.scheduler import CosineWarmupScheduler
from app.ml.training.checkpointing import save_training_checkpoint, load_training_checkpoint
from app.ml.training.evaluator import Evaluator
from app.ml.training.metrics import MetricsTracker
from app.ml.training.trainer import PretrainingEngine

__all__ = [
    "TrainingConfig",
    "TrainingError",
    "DataLoadingError",
    "NumericalInstabilityError",
    "DeviceMismatchError",
    "TrainingInterruptedError",
    "set_seed",
    "TokenizedDataset",
    "create_dataloader",
    "configure_optimizer",
    "CosineWarmupScheduler",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "Evaluator",
    "MetricsTracker",
    "PretrainingEngine",
]
