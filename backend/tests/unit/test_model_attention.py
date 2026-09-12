"""
Unit tests for Multi-Head Causal Self-Attention.
"""

import pytest
import torch

from app.ml.model import ModelConfig, MultiHeadCausalAttention
from app.ml.model.exceptions import ModelConfigError


def test_attention_shapes_and_output():
    config = ModelConfig(d_model=128, n_heads=4, attention_dropout=0.0, dropout=0.0)
    attn = MultiHeadCausalAttention(config)

    x = torch.randn(2, 8, 128)  # Batch 2, Seq 8, Dim 128
    out = attn(x)

    assert out.shape == (2, 8, 128)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_attention_indivisible_heads():
    config = ModelConfig(d_model=100, n_heads=3)
    with pytest.raises(ModelConfigError):
        _ = config.d_k
