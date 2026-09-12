"""
Precious Edu LLM — Parameter Initialization

Deliberate initialization for Transformer module parameters:
- Normal distribution (mean=0.0, std=0.02) for Linear & Embedding weights
- Zero initialization for Linear biases
- One initialization for LayerNorm weights, zero for LayerNorm biases
"""

import torch
import torch.nn as nn


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """
    Recursively initialize weights of a PyTorch Module.

    Args:
        module: PyTorch module instance.
        std: Standard deviation for normal weight initialization.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0.0)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        if module.weight is not None:
            torch.nn.init.ones_(module.weight)
