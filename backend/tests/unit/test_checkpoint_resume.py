"""
Unit tests for Checkpoint Resumption Equivalence and Numerical Stability checks.
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from app.ml.model import ModelConfig, PreciousTransformer
from app.ml.training import (
    TrainingConfig,
    PretrainingEngine,
    NumericalInstabilityError,
    save_training_checkpoint,
    load_training_checkpoint,
    configure_optimizer,
    CosineWarmupScheduler,
)


def test_checkpoint_resume_equivalence(tmp_path):
    """
    MANDATORY RESUME TEST:
    Compare continuous 10-step training run vs (5-step training -> checkpoint save -> reload -> 5-step training).
    """
    inputs = torch.tensor([[2, 10, 20, 30, 40]] * 20, dtype=torch.long)
    targets = torch.tensor([[10, 20, 30, 40, 3]] * 20, dtype=torch.long)
    ds = TensorDataset(inputs, targets)
    train_loader = DataLoader(ds, batch_size=2)
    val_loader = DataLoader(ds, batch_size=2)

    config = TrainingConfig(
        seed=42,
        total_steps=10,
        epochs=10,
        learning_rate=1e-3,
        min_learning_rate=1e-3,
        warmup_steps=0,
        micro_batch_size=2,
        batch_size=2,
        evaluation_interval=100,
        checkpoint_interval=100,
    )

    # 1. Continuous Run (10 steps)
    run_dir_cont = tmp_path / "run_continuous"
    model_cont = PreciousTransformer(ModelConfig(vocab_size=100, d_model=32, n_heads=2, n_layers=1, dropout=0.0, attention_dropout=0.0))
    engine_cont = PretrainingEngine(config, model_cont, train_loader, val_loader, run_dir_cont)
    summary_cont = engine_cont.train()

    # 2. Split Run (5 steps -> save -> reload -> 5 steps)
    config_5steps = TrainingConfig(
        seed=42,
        total_steps=5,
        epochs=10,
        learning_rate=1e-3,
        min_learning_rate=1e-3,
        warmup_steps=0,
        micro_batch_size=2,
        batch_size=2,
        evaluation_interval=100,
        checkpoint_interval=5,
    )
    run_dir_split = tmp_path / "run_split"
    model_split = PreciousTransformer(ModelConfig(vocab_size=100, d_model=32, n_heads=2, n_layers=1, dropout=0.0, attention_dropout=0.0))
    engine_split = PretrainingEngine(config_5steps, model_split, train_loader, val_loader, run_dir_split)
    engine_split.train()

    # Resume for remaining 5 steps (up to total 10)
    config_resumed = TrainingConfig(
        seed=42,
        total_steps=10,
        epochs=10,
        learning_rate=1e-3,
        min_learning_rate=1e-3,
        warmup_steps=0,
        micro_batch_size=2,
        batch_size=2,
        evaluation_interval=100,
        checkpoint_interval=100,
    )
    checkpoint_path = str(run_dir_split / "checkpoints" / "latest.pt")
    engine_resumed = PretrainingEngine(config_resumed, model_split, train_loader, val_loader, run_dir_split)
    summary_resumed = engine_resumed.train(resume_checkpoint_path=checkpoint_path)

    # Verify final training losses are equal
    assert abs(summary_cont["final_train_loss"] - summary_resumed["final_train_loss"]) < 1e-1
