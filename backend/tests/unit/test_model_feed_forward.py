"""
Unit tests for FeedForward and TransformerBlock modules.
"""

import pytest
import torch

from app.ml.model import ModelConfig, FeedForward, TransformerBlock


def test_feed_forward_shape():
    config = ModelConfig(d_model=64, d_ff=256, activation="gelu", dropout=0.0)
    ffn = FeedForward(config)

    x = torch.randn(3, 10, 64)
    out = ffn(x)

    assert out.shape == (3, 10, 64)
    assert not torch.isnan(out).any()


def test_transformer_block_pre_ln_and_residual():
    config = ModelConfig(d_model=64, n_heads=2, d_ff=128, dropout=0.0, attention_dropout=0.0)
    block = TransformerBlock(config)

    x = torch.randn(2, 5, 64)
    out = block(x)

    assert out.shape == (2, 5, 64)
    assert not torch.isnan(out).any()
