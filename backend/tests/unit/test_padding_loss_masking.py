"""
Unit tests for Padding Loss Masking (ignore_index = pad_token_id).
"""

import pytest
import torch

from app.ml.model import ModelConfig, PreciousTransformer


def test_padding_loss_masking():
    """
    Verify that altering padded target token IDs (<pad>: 0) does NOT alter calculated loss.
    """
    config = ModelConfig(vocab_size=100, pad_token_id=0, d_model=32, n_heads=2, n_layers=1, dropout=0.0)
    model = PreciousTransformer(config)
    model.eval()

    input_ids = torch.tensor([[10, 20, 30, 0, 0]], dtype=torch.long)
    # Targets with pad_token_id = 0 at position 3 and 4
    targets_orig = torch.tensor([[20, 30, 0, 0, 0]], dtype=torch.long)
    # Targets with pad_token_id = 0 at position 3 and 4 (same pad index)
    targets_alt = torch.tensor([[20, 30, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        _, loss1 = model(input_ids, targets=targets_orig)
        _, loss2 = model(input_ids, targets=targets_alt)

    assert torch.allclose(loss1, loss2)
