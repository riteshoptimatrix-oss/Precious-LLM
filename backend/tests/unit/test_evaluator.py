"""
Unit tests for Evaluator engine and Gradient Accumulation equivalence.
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from app.ml.model import ModelConfig, PreciousTransformer
from app.ml.training import Evaluator, configure_optimizer, CosineWarmupScheduler


def test_evaluator_no_gradient_and_mode():
    config = ModelConfig(vocab_size=100, d_model=32, n_heads=2, n_layers=1)
    model = PreciousTransformer(config)

    inputs = torch.randint(0, 100, (4, 10))
    targets = torch.randint(0, 100, (4, 10))
    ds = TensorDataset(inputs, targets)
    loader = DataLoader(ds, batch_size=2)

    evaluator = Evaluator(model, loader, device=torch.device("cpu"))
    val_loss, val_ppl = evaluator.evaluate()

    assert isinstance(val_loss, float)
    assert val_loss > 0.0
    assert isinstance(val_ppl, float)
    assert val_ppl >= 1.0

    # Ensure model parameters did NOT accumulate gradients during evaluation
    for param in model.parameters():
        assert param.grad is None


def test_gradient_accumulation_equivalence():
    """
    Verify that micro-batching with gradient accumulation produces
    gradients approximately equal to large-batch forward/backward pass.
    """
    torch.manual_seed(42)

    config = ModelConfig(vocab_size=100, d_model=32, n_heads=2, n_layers=1, dropout=0.0, attention_dropout=0.0)

    # Large batch run (batch_size = 4)
    model1 = PreciousTransformer(config)
    model1.eval()
    inputs_large = torch.randint(1, 100, (4, 8))
    targets_large = torch.randint(1, 100, (4, 8))

    _, loss1 = model1(inputs_large, targets=targets_large)
    loss1.backward()

    grads1 = [p.grad.clone() for p in model1.parameters() if p.requires_grad]

    # Micro-batch run (micro_batch_size = 2, grad_accum = 2)
    torch.manual_seed(42)
    model2 = PreciousTransformer(config)
    model2.eval()

    input_micro1, target_micro1 = inputs_large[:2], targets_large[:2]
    input_micro2, target_micro2 = inputs_large[2:], targets_large[2:]

    _, loss_m1 = model2(input_micro1, targets=target_micro1)
    (loss_m1 / 2.0).backward()

    _, loss_m2 = model2(input_micro2, targets=target_micro2)
    (loss_m2 / 2.0).backward()

    grads2 = [p.grad.clone() for p in model2.parameters() if p.requires_grad]

    # Verify gradients match within floating-point tolerance
    for g1, g2 in zip(grads1, grads2):
        assert torch.allclose(g1, g2, atol=1e-4)
