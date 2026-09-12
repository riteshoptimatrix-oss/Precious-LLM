"""
Unit tests for full PreciousTransformer model.
"""

import pytest
import torch

from app.ml.model import ModelConfig, PreciousTransformer
from app.ml.model.exceptions import ContextLengthExceededError


def test_transformer_forward_logits_and_loss():
    config = ModelConfig(
        vocab_size=175,
        max_seq_length=128,
        d_model=64,
        n_heads=2,
        n_layers=2,
        weight_tying=True,
    )
    model = PreciousTransformer(config)

    input_ids = torch.tensor([[1, 5, 10, 20], [2, 6, 12, 24]], dtype=torch.long)
    targets = torch.tensor([[5, 10, 20, 3], [6, 12, 24, 3]], dtype=torch.long)

    logits, loss = model(input_ids, targets=targets)

    assert logits.shape == (2, 4, 175)
    assert isinstance(loss.item(), float)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_weight_tying():
    config = ModelConfig(vocab_size=175, d_model=64, weight_tying=True)
    model = PreciousTransformer(config)

    assert model.lm_head.weight is model.embedding.token_embedding.embedding.weight


def test_no_weight_tying():
    config = ModelConfig(vocab_size=175, d_model=64, weight_tying=False)
    model = PreciousTransformer(config)

    assert model.lm_head.weight is not model.embedding.token_embedding.embedding.weight


def test_batch_independence():
    """Verify item 0 logits are completely independent of item 1 in batch."""
    config = ModelConfig(vocab_size=100, max_seq_length=64, d_model=32, n_heads=2, n_layers=1, dropout=0.0)
    model = PreciousTransformer(config)
    model.eval()

    sample_a = torch.tensor([[1, 2, 3]], dtype=torch.long)
    sample_b1 = torch.tensor([[4, 5, 6]], dtype=torch.long)
    sample_b2 = torch.tensor([[7, 8, 9]], dtype=torch.long)

    batch1 = torch.cat([sample_a, sample_b1], dim=0)  # Shape [2, 3]
    batch2 = torch.cat([sample_a, sample_b2], dim=0)  # Shape [2, 3]

    with torch.no_grad():
        logits1 = model(batch1)
        logits2 = model(batch2)

    # Item 0 in batch 1 and item 0 in batch 2 should produce exact same logits
    assert torch.allclose(logits1[0], logits2[0], atol=1e-5)
