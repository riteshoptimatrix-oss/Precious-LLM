"""
Precious Edu LLM — Pretraining Configuration

Dataclass defining hyperparameters, directory paths, optimizer settings, scheduler bounds,
batching rules, and device placement for LLM pretraining.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os
import yaml

from app.ml.training.exceptions import TrainingError


@dataclass
class TrainingConfig:
    """
    Configuration settings governing the LLM pretraining pipeline.
    """

    # Seed & Device
    seed: int = 42
    device: str = "auto"               # "auto", "cpu", "cuda"

    # Batching & Accumulation
    batch_size: int = 4                # Effective batch size
    micro_batch_size: int = 2          # Per-forward step batch size
    gradient_accumulation_steps: int = 2 # Calculated automatically if micro_batch_size specified

    # Optimizer (AdamW)
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    gradient_clip_norm: float = 1.0

    # Scheduler & Steps
    warmup_steps: int = 100
    total_steps: int = 1000
    epochs: int = 10

    # Interval Frequencies
    evaluation_interval: int = 50
    checkpoint_interval: int = 100
    logging_interval: int = 10

    # Data Bounds
    max_sequence_length: int = 512
    num_workers: int = 0
    mixed_precision: str = "fp32"      # "fp32", "fp16", "bf16"

    # Paths
    dataset_dir: str = "data/processed/tokenized"
    artifacts_dir: str = "artifacts/training"

    def __post_init__(self):
        """Auto-calculate gradient accumulation steps if batch sizes align."""
        if self.micro_batch_size > 0 and self.batch_size > 0:
            if self.batch_size % self.micro_batch_size == 0:
                self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size

    def validate(self) -> None:
        """Validate configuration hyperparameter consistency."""
        if self.seed < 0:
            raise TrainingError(f"seed must be >= 0, got {self.seed}")
        if self.learning_rate <= 0.0:
            raise TrainingError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.min_learning_rate < 0.0 or self.min_learning_rate > self.learning_rate:
            raise TrainingError(f"min_learning_rate must be between 0.0 and learning_rate ({self.learning_rate})")
        if self.micro_batch_size <= 0:
            raise TrainingError(f"micro_batch_size must be > 0, got {self.micro_batch_size}")
        if self.batch_size <= 0:
            raise TrainingError(f"batch_size must be > 0, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise TrainingError(f"gradient_accumulation_steps must be > 0")
        if self.max_sequence_length <= 0:
            raise TrainingError(f"max_sequence_length must be > 0, got {self.max_sequence_length}")
        if self.gradient_clip_norm < 0.0:
            raise TrainingError(f"gradient_clip_norm must be >= 0, got {self.gradient_clip_norm}")
        if self.mixed_precision not in ("fp32", "fp16", "bf16"):
            raise TrainingError(f"mixed_precision must be 'fp32', 'fp16', or 'bf16'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Construct configuration from dictionary and validate."""
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        if not os.path.exists(path):
            raise TrainingError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "training" in data:
            data = data["training"]
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"training": self.to_dict()}, f, default_flow_style=False, sort_keys=False)
