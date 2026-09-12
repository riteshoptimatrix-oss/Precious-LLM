"""
Precious Edu LLM — Cosine Warmup Learning Rate Scheduler

Implements linear warmup followed by cosine decay down to minimum learning rate.
Fully deterministic, saveable, and restorable.
"""

import math
from typing import List, Dict, Any
import torch
from torch.optim import Optimizer


class CosineWarmupScheduler:
    """
    Learning Rate Scheduler with linear warmup and cosine decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        min_learning_rate: float = 5e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_learning_rate = min_learning_rate
        self.base_lrs: List[float] = [group["lr"] for group in optimizer.param_groups]
        self.current_step: int = 0

    def get_lr(self) -> List[float]:
        """Calculate current learning rate for each parameter group."""
        if self.current_step < self.warmup_steps and self.warmup_steps > 0:
            # Linear warmup: scale from 0.0 up to base_lr
            lr_scale = float(self.current_step) / float(self.warmup_steps)
            return [base_lr * lr_scale for base_lr in self.base_lrs]

        if self.current_step >= self.total_steps:
            # After total_steps: hold at min_learning_rate
            return [self.min_learning_rate for _ in self.base_lrs]

        # Cosine decay phase between warmup_steps and total_steps
        decay_steps = self.total_steps - self.warmup_steps
        step_in_decay = self.current_step - self.warmup_steps
        progress = float(step_in_decay) / float(max(1, decay_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        lrs = []
        for base_lr in self.base_lrs:
            min_lr = min(self.min_learning_rate, base_lr)
            lr = min_lr + (base_lr - min_lr) * cosine_decay
            lrs.append(lr)
        return lrs

    def step(self) -> None:
        """Advance scheduler by one step and update optimizer learning rates."""
        self.current_step += 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary for serialization."""
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_learning_rate": self.min_learning_rate,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from state dictionary."""
        self.current_step = state_dict.get("current_step", 0)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.total_steps = state_dict.get("total_steps", self.total_steps)
        self.min_learning_rate = state_dict.get("min_learning_rate", self.min_learning_rate)
        self.base_lrs = state_dict.get("base_lrs", self.base_lrs)
        
        # Apply updated learning rate immediately to optimizer
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
