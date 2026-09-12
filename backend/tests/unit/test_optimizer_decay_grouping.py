"""
Unit tests for Optimizer Weight Decay Grouping and CosineWarmupScheduler.
"""

import pytest
import torch

from app.ml.model import ModelConfig, PreciousTransformer
from app.ml.training import configure_optimizer, CosineWarmupScheduler


def test_optimizer_parameter_decay_grouping():
    config = ModelConfig(vocab_size=100, d_model=32, n_heads=2, n_layers=1)
    model = PreciousTransformer(config)

    optimizer = configure_optimizer(model, learning_rate=1e-3, weight_decay=0.1)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

    # Ensure biases and LayerNorm weights are in group 1 (weight_decay == 0.0)
    decay_params = set(optimizer.param_groups[0]["params"])
    no_decay_params = set(optimizer.param_groups[1]["params"])

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "ln" in name:
            assert param in no_decay_params
        else:
            assert param in decay_params


def test_cosine_warmup_scheduler():
    config = ModelConfig(vocab_size=100, d_model=32, n_heads=2, n_layers=1)
    model = PreciousTransformer(config)
    optimizer = configure_optimizer(model, learning_rate=1e-3)

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
        min_learning_rate=1e-4,
    )

    # Warmup phase: step 0 -> 10 should increase LR
    lr_initial = scheduler.get_lr()[0]
    for _ in range(5):
        scheduler.step()
    lr_mid_warmup = scheduler.get_lr()[0]

    assert lr_mid_warmup > lr_initial

    # Step to end of warmup
    for _ in range(5):
        scheduler.step()
    lr_peak = scheduler.get_lr()[0]
    assert torch.isclose(torch.tensor(lr_peak), torch.tensor(1e-3), atol=1e-5)

    # Cosine decay phase: step 10 -> 100 should decrease LR
    for _ in range(50):
        scheduler.step()
    lr_decay = scheduler.get_lr()[0]
    assert lr_decay < lr_peak
