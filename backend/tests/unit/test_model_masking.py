"""
Unit tests for Causal Masking and Future-Token Leakage Prevention.
"""

import pytest
import torch

from app.ml.model import ModelConfig, PreciousTransformer, create_causal_mask


def test_create_causal_mask_structure():
    seq_len = 4
    mask = create_causal_mask(seq_len)

    assert mask.shape == (1, 1, 4, 4)

    # Check lower triangular structure: 0.0 for visible, -inf for future
    m = mask[0, 0]
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                assert m[i, j].item() == 0.0
            else:
                assert m[i, j].item() == float("-inf")


def test_future_token_leakage_prevention():
    """
    MANDATORY REQUIREMENT:
    Verify that sequence position i logits depend ONLY on tokens at position <= i,
    and are completely unchanged when future tokens at j > i change.
    """
    torch.manual_seed(42)

    config = ModelConfig(
        vocab_size=100,
        max_seq_length=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model = PreciousTransformer(config)
    model.eval()

    # Sequence A: [A, B, C, X]
    seq_a = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    # Sequence B: [A, B, C, Y]  (Same prefix [10, 20, 30], different 4th token)
    seq_b = torch.tensor([[10, 20, 30, 99]], dtype=torch.long)

    with torch.no_grad():
        logits_a = model(seq_a)  # [1, 4, 100]
        logits_b = model(seq_b)  # [1, 4, 100]

    # Predictions for positions 0, 1, 2 (corresponding to A, B, C) MUST be IDENTICAL
    assert torch.allclose(logits_a[:, 0, :], logits_b[:, 0, :], atol=1e-5), \
        "Position 0 logits changed when position 3 token changed!"
    assert torch.allclose(logits_a[:, 1, :], logits_b[:, 1, :], atol=1e-5), \
        "Position 1 logits changed when position 3 token changed!"
    assert torch.allclose(logits_a[:, 2, :], logits_b[:, 2, :], atol=1e-5), \
        "Position 2 logits changed when position 3 token changed!"

    # Position 3 logits can differ because input at position 3 differed
    assert not torch.allclose(logits_a[:, 3, :], logits_b[:, 3, :], atol=1e-5)
