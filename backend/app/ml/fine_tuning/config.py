"""
Precious Edu LLM — Fine-Tuning Configuration

Dataclass governing hyperparameters, paths, loss masking settings, evaluation intervals,
parameter freezing options, and generation defaults for conversational fine-tuning.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os
import yaml
from app.ml.fine_tuning.exceptions import FineTuningError


@dataclass
class FineTuningConfig:
    """
    Configuration settings governing Phase 8 Conversational Fine-Tuning.
    """

    # Seed & Device
    seed: int = 42
    device: str = "auto"               # "auto", "cpu", "cuda"

    # Batching & Accumulation
    batch_size: int = 4                # Effective total batch size
    micro_batch_size: int = 2          # Per-forward micro batch size
    gradient_accumulation_steps: int = 2 # Calculated automatically if micro_batch_size specified

    # Optimizer (AdamW)
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    gradient_clip_norm: float = 1.0

    # Scheduler & Steps
    warmup_steps: int = 10
    max_steps: int = 200
    epochs: int = 5

    # Loss Masking
    ignore_index: int = -100
    mask_user_prompt: bool = True

    # Freezing Options
    freeze_strategy: str = "none"      # "none", "embeddings", "first_n", "all_except_head"
    freeze_first_n_layers: int = 0

    # Interval Frequencies
    evaluation_interval: int = 10
    checkpoint_interval: int = 20
    logging_interval: int = 5

    # Data Bounds
    max_sequence_length: int = 256
    num_workers: int = 0
    mixed_precision: str = "fp32"      # "fp32", "fp16", "bf16"

    # Inference / Evaluation Default Params
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 64
    repetition_penalty: float = 1.1

    # Paths
    pretrained_checkpoint_path: str = "artifacts/training/run-train_model/checkpoints/latest.pt"
    tokenizer_dir: str = "artifacts/tokenizer/v1"
    dataset_dir: str = "data/training/conversational"
    artifacts_dir: str = "artifacts/fine_tuning"

    def __post_init__(self):
        """Auto-calculate gradient accumulation steps if batch sizes align."""
        if self.micro_batch_size > 0 and self.batch_size > 0:
            if self.batch_size % self.micro_batch_size == 0:
                self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size

    def validate(self) -> None:
        """Validate hyperparameter consistency."""
        if self.seed < 0:
            raise FineTuningError(f"seed must be >= 0, got {self.seed}")
        if self.learning_rate <= 0.0:
            raise FineTuningError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.min_learning_rate < 0.0 or self.min_learning_rate > self.learning_rate:
            raise FineTuningError(f"min_learning_rate must be between 0.0 and learning_rate ({self.learning_rate})")
        if self.micro_batch_size <= 0:
            raise FineTuningError(f"micro_batch_size must be > 0, got {self.micro_batch_size}")
        if self.batch_size <= 0:
            raise FineTuningError(f"batch_size must be > 0, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise FineTuningError(f"gradient_accumulation_steps must be > 0")
        if self.max_sequence_length <= 0:
            raise FineTuningError(f"max_sequence_length must be > 0, got {self.max_sequence_length}")
        if self.freeze_strategy not in ("none", "embeddings", "first_n", "all_except_head"):
            raise FineTuningError(f"Invalid freeze_strategy: {self.freeze_strategy}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FineTuningConfig":
        """Construct configuration from dictionary and validate."""
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "FineTuningConfig":
        """Load configuration from YAML file."""
        if not os.path.exists(path):
            raise FineTuningError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "fine_tuning" in data:
            data = data["fine_tuning"]
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"fine_tuning": self.to_dict()}, f, default_flow_style=False, sort_keys=False)


def get_finetuning_config() -> FineTuningConfig:
    """Get default FineTuningConfig instance."""
    config = FineTuningConfig()
    config.validate()
    return config
