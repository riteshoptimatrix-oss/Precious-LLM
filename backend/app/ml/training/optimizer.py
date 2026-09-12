"""
Precious Edu LLM — Decoupled Weight Decay AdamW Optimizer

Constructs AdamW optimizer with parameter group separation:
- 2D parameters (weights of Linear and Embedding layers) receive weight_decay
- 1D parameters (biases, LayerNorm weights & biases) are excluded from weight_decay
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any

from app.ml.model import PreciousTransformer


def configure_optimizer(
    model: PreciousTransformer,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Configure AdamW optimizer with parameter group separation for weight decay.

    Args:
        model: PreciousTransformer instance.
        learning_rate: Initial learning rate.
        weight_decay: L2 penalty weight decay factor.
        betas: Adam beta parameters (beta1, beta2).
        eps: Small constant for numerical stability.

    Returns:
        Configured AdamW optimizer instance.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 1D parameters (biases, LayerNorm weights/biases) do not receive weight decay
        if param.ndim < 2 or "bias" in name or "ln" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )
    return optimizer
