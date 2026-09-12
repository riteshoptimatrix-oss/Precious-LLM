"""
Unit tests for numerical stability detection and full CPU pipeline integration.
"""

import pytest
import json
import torch
from pathlib import Path

from app.ml.model import ModelConfig, PreciousTransformer
from app.ml.training import (
    TrainingConfig,
    PretrainingEngine,
    TokenizedDataset,
    create_dataloader,
    NumericalInstabilityError,
)


def test_nan_loss_detection(tmp_path):
    """
    Verify PretrainingEngine detects NaN loss and raises NumericalInstabilityError.
    """
    jsonl_path = tmp_path / "nan_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"input_ids": [2, 10, 20, 30, 40, 50, 3]}) + "\n")

    dataset = TokenizedDataset(jsonl_path=jsonl_path, max_seq_len=6, pad_token_id=0, vocab_size=100)
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    val_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    config = ModelConfig(vocab_size=100, max_seq_length=6, d_model=32, n_heads=2, n_layers=1)
    model = PreciousTransformer(config)

    # Artificially inject NaN into embedding weight
    with torch.no_grad():
        model.embedding.token_embedding.embedding.weight[0, 0] = float("nan")

    t_config = TrainingConfig(total_steps=5, micro_batch_size=2, batch_size=2, artifacts_dir=str(tmp_path))
    engine = PretrainingEngine(t_config, model, train_loader, val_loader, tmp_path / "run_nan")

    with pytest.raises(NumericalInstabilityError):
        engine.train()


def test_full_cpu_pretraining_pipeline_integration(tmp_path):
    """
    MANDATORY CPU PIPELINE INTEGRATION TEST:
    Executes full pipeline end-to-end: TokenizedDataset -> DataLoader -> Model -> PretrainingEngine -> Checkpoint -> Evaluation.
    """
    # Create temporary dataset file
    jsonl_path = tmp_path / "train_tokenized.jsonl"
    data = [
        {"input_ids": [2, 10, 20, 30, 40, 50, 3]},
        {"input_ids": [2, 15, 25, 35, 45, 55, 3]},
        {"input_ids": [2, 12, 22, 32, 42, 52, 3]},
        {"input_ids": [2, 18, 28, 38, 48, 58, 3]},
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    dataset = TokenizedDataset(jsonl_path=jsonl_path, max_seq_len=6, pad_token_id=0, vocab_size=100)
    train_loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    val_loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    model_config = ModelConfig(vocab_size=100, max_seq_length=6, d_model=32, n_heads=2, n_layers=1, dropout=0.0, attention_dropout=0.0)
    model = PreciousTransformer(model_config)

    t_config = TrainingConfig(
        seed=42,
        device="cpu",
        total_steps=20,
        epochs=10,
        learning_rate=5e-3,
        warmup_steps=0,
        micro_batch_size=2,
        batch_size=2,
        evaluation_interval=10,
        checkpoint_interval=10,
        artifacts_dir=str(tmp_path),
    )

    run_dir = tmp_path / "run_integration"
    engine = PretrainingEngine(t_config, model, train_loader, val_loader, run_dir)

    # 1. Execute Dry Run
    dry_result = engine.dry_run()
    assert dry_result["status"] == "PASS"

    # 2. Execute Training
    summary = engine.train()
    assert summary["status"] == "COMPLETED"
    assert summary["total_steps"] == 20
    assert summary["final_train_loss"] < summary["initial_train_loss"]

    # 3. Verify Checkpoints & Manifest Files Created
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
