"""
Unit tests for PreciousTransformer model determinism.
"""

import pytest
import torch

from app.ml.model import ModelConfig, PreciousTransformer


def test_model_determinism_under_fixed_seed():
    config = ModelConfig(vocab_size=175, d_model=64, n_heads=2, n_layers=2, dropout=0.0)

    # Instantiate Model 1 with seed 42
    torch.manual_seed(42)
    model1 = PreciousTransformer(config)

    # Instantiate Model 2 with seed 42
    torch.manual_seed(42)
    model2 = PreciousTransformer(config)

    # Verify parameters match exactly
    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2
        assert torch.equal(p1, p2), f"Parameter mismatch for {name1}"

    # Verify forward pass outputs match
    input_ids = torch.tensor([[1, 10, 20, 30]], dtype=torch.long)
    model1.eval()
    model2.eval()

    with torch.no_grad():
        out1 = model1(input_ids)
        out2 = model2(input_ids)

    assert torch.equal(out1, out2)
