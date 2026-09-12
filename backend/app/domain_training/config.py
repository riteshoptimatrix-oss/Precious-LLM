"""
Precious Edu LLM — Domain Training Configuration

Defines hyperparameter configuration for supervised domain fine-tuning.
Start checkpoint: Phase 8 fine-tuned model checkpoint.
Tokenizer: Phase 5 custom tokenizer.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


def _resolve_project_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    if p.exists():
        return p.resolve()
    # Check project root (parent of app directory)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    p2 = project_root / rel_or_abs
    if p2.exists():
        return p2.resolve()
    return p.resolve()


class DomainTrainingConfig(BaseModel):
    """Configuration for Phase 11 domain fine-tuning."""
    # Experiment metadata
    run_name: str = Field("domain_finetune", description="Name prefix for training run")
    base_model_checkpoint: str = Field(
        "artifacts/fine_tuning/run-finetune-20260912-132935/final/model.pt",
        description="Path to starting Phase 8 model checkpoint"
    )
    tokenizer_path: str = Field("artifacts/tokenizer/v1", description="Path to Phase 5 tokenizer directory")
    data_root: str = Field("data/domain", description="Root directory for domain datasets")
    output_dir: str = Field("artifacts/domain_training", description="Directory to store run outputs")

    # Hyperparameters
    seed: int = Field(42, description="Random seed for reproducibility")
    device: str = Field("auto", description="Device (cuda/cpu/auto)")
    epochs: int = Field(5, description="Total training epochs")
    batch_size: int = Field(2, description="Per-step batch size")
    micro_batch_size: int = Field(1, description="Micro-batch size for gradient accumulation")
    gradient_accumulation_steps: int = Field(2, description="Gradient accumulation steps")

    learning_rate: float = Field(5e-5, description="Initial learning rate")
    min_learning_rate: float = Field(5e-6, description="Minimum learning rate")
    weight_decay: float = Field(0.01, description="AdamW weight decay")
    beta1: float = Field(0.9, description="AdamW beta1")
    beta2: float = Field(0.95, description="AdamW beta2")
    eps: float = Field(1e-8, description="AdamW epsilon")
    max_grad_norm: float = Field(1.0, description="Gradient clipping norm limit")

    warmup_steps: int = Field(5, description="Linear warmup steps")
    max_sequence_length: int = Field(64, description="Maximum sequence length")

    # Mixed Dataset Ratios (Prevents Catastrophic Forgetting)
    domain_ratio: float = Field(0.5, description="Proportion of domain training examples")
    general_conversation_ratio: float = Field(0.3, description="Proportion of general conversation examples")
    instruction_ratio: float = Field(0.2, description="Proportion of instruction examples")

    # Logging & Checkpointing
    log_interval: int = Field(1, description="Steps between log outputs")
    eval_interval: int = Field(2, description="Steps between evaluation passes")
    checkpoint_interval: int = Field(5, description="Steps between saving checkpoints")

    @property
    def resolved_base_checkpoint(self) -> Path:
        return _resolve_project_path(self.base_model_checkpoint)

    @property
    def resolved_tokenizer_dir(self) -> Path:
        return _resolve_project_path(self.tokenizer_path)

    @property
    def resolved_data_root(self) -> Path:
        return _resolve_project_path(self.data_root)

    @property
    def resolved_output_dir(self) -> Path:
        return _resolve_project_path(self.output_dir)

    def get_run_dir(self, timestamp_str: str) -> Path:
        """Compute output directory path for a run."""
        run_dir = self.resolved_output_dir / f"run-{self.run_name}-{timestamp_str}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
