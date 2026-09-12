"""
Precious Edu LLM — Multi-Head Causal Self-Attention

Implements multi-head causal self-attention mechanism from scratch using PyTorch primitives.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from app.ml.model.config import ModelConfig
from app.ml.model.masking import create_causal_mask


class MultiHeadCausalAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention block.

    Calculates scaled dot-product attention with causal lower-triangular masking across heads.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k  # d_model // n_heads

        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, d_model].
            attention_mask: Optional additional mask tensor [B, 1, T, T] or [1, 1, T, T].

        Returns:
            Tensor of shape [B, T, d_model].
        """
        batch_size, seq_len, d_model = x.shape

        # 1. Project Q, K, V: [B, T, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape & transpose for multi-head attention: [B, n_heads, T, d_k]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention scores: [B, n_heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. Apply Causal Mask: [1, 1, T, T]
        causal_mask = create_causal_mask(seq_len, device=x.device, dtype=x.dtype)
        scores = scores + causal_mask

        # 5. Apply optional external mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Convert 2D binary mask [B, T] to 4D additive mask [B, 1, 1, T]
                attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=x.dtype)) * -10000.0
            scores = scores + attention_mask

        # 6. Softmax over last dimension (keys/positions)
        weights = torch.softmax(scores, dim=-1)

        # 7. Apply attention dropout
        weights = self.attn_dropout(weights)

        # 8. Weighted sum of values: [B, n_heads, T, d_k]
        attn_out = torch.matmul(weights, v)

        # 9. Concatenate / transpose heads back: [B, T, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 10. Output projection & residual dropout
        output = self.out_proj(attn_out)
        return self.resid_dropout(output)
