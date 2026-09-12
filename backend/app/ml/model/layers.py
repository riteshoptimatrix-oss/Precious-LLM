"""
Precious Edu LLM — Feed-Forward & Transformer Decoder Block

Implements FeedForward network and Pre-LN TransformerBlock.
"""

import torch
import torch.nn as nn
from typing import Optional

from app.ml.model.config import ModelConfig
from app.ml.model.attention import MultiHeadCausalAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    Architecture:
    Linear(d_model -> d_ff) -> GELU/ReLU -> Linear(d_ff -> d_model) -> Dropout
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        
        if config.activation == "gelu":
            self.act = nn.GELU()
        elif config.activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {config.activation}")

        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, d_model].

        Returns:
            Tensor of shape [B, T, d_model].
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Decoder-only Pre-LayerNorm Transformer Block.

    Structure:
    Input -> LN1 -> Causal Self-Attention -> (+) Residual -> LN2 -> FFN -> (+) Residual -> Output
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = MultiHeadCausalAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, d_model].
            attention_mask: Optional additional mask tensor.

        Returns:
            Tensor of shape [B, T, d_model].
        """
        # Pre-LN Causal Self-Attention with Residual Connection
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)

        # Pre-LN Feed-Forward Network with Residual Connection
        x = x + self.ffn(self.ln_2(x))

        return x
