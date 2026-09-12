"""
Unit tests for tiny model training smoke test, backpropagation, gradients, and parameter updates.
"""

import pytest
import torch

from app.ml.model import ModelConfig, PreciousTransformer


def test_tiny_training_smoke_test():
    """
    REQUIRED SMOKE TEST:
    Verify forward pass, loss calculation, backward pass, gradient existence,
    parameter update, and loss reduction over a micro synthetic dataset.
    """
    torch.manual_seed(42)

    config = ModelConfig(
        vocab_size=100,
        max_seq_length=32,
        d_model=32,
        n_heads=2,
        n_layers=2,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model = PreciousTransformer(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    # Micro synthetic dataset (repeat same sequence)
    # Inputs:  [2, 10, 20, 30]
    # Targets: [10, 20, 30, 3]
    inputs = torch.tensor([[2, 10, 20, 30]] * 4, dtype=torch.long)
    targets = torch.tensor([[10, 20, 30, 3]] * 4, dtype=torch.long)

    # Step 1: Compute initial loss
    logits, initial_loss = model(inputs, targets=targets)
    initial_loss_val = initial_loss.item()

    # Step 2: Record initial parameter weights
    initial_param_weight = model.blocks[0].attn.q_proj.weight.clone()

    # Step 3: Run optimization steps
    losses = []
    for step in range(30):
        optimizer.zero_grad()
        logits, loss = model(inputs, targets=targets)
        loss.backward()

        # Verify gradient existence & finiteness
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

        optimizer.step()
        losses.append(loss.item())

    final_loss_val = losses[-1]

    # Verify parameters updated
    updated_param_weight = model.blocks[0].attn.q_proj.weight.clone()
    assert not torch.equal(initial_param_weight, updated_param_weight), \
        "Model parameters did not update after optimizer.step()!"

    # Verify loss decreased substantially
    assert final_loss_val < initial_loss_val, \
        f"Initial loss ({initial_loss_val:.4f}) should be > final loss ({final_loss_val:.4f})"
