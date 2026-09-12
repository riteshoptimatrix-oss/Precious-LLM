"""
Unit tests for FineTuningEngine and Checkpoint system in app.ml.fine_tuning.trainer
"""

import pytest
import json
from pathlib import Path
import torch

from app.ml.fine_tuning.config import FineTuningConfig
from app.ml.fine_tuning.dataset import ConversationalFineTuningDataset
from app.ml.fine_tuning.trainer import FineTuningEngine
from app.tokenizer.tokenizer import Tokenizer


def resolve_artifact(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p.resolve()
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    p2 = project_root / path_str
    if p2.exists():
        return p2.resolve()
    p3 = Path("..") / path_str
    if p3.exists():
        return p3.resolve()
    return p


@pytest.fixture
def tokenizer():
    tok_dir = resolve_artifact("artifacts/tokenizer/v1")
    return Tokenizer.load(tok_dir)


def test_finetuning_trainer_dry_run(tmp_path, tokenizer):
    data = [
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello!"}]},
        {"messages": [{"role": "user", "content": "Thanks"}, {"role": "assistant", "content": "You are welcome!"}]},
    ]
    data_file = tmp_path / "train.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    ckpt_path = resolve_artifact("artifacts/fine_tuning/run-finetune-20260912-132935/final/model.pt")
    if not ckpt_path.exists():
        ckpt_path = resolve_artifact("artifacts/training/run-train_model/checkpoints/latest.pt")

    config = FineTuningConfig(
        max_steps=2,
        batch_size=2,
        micro_batch_size=2,
        epochs=2,
        evaluation_interval=1,
        checkpoint_interval=2,
        max_sequence_length=32,
        artifacts_dir=str(tmp_path / "artifacts"),
        pretrained_checkpoint_path=str(ckpt_path),
    )

    dataset = ConversationalFineTuningDataset(
        jsonl_path=str(data_file),
        tokenizer=tokenizer,
        config=config,
    )

    engine = FineTuningEngine(
        config=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        val_dataset=dataset,
    )

    assert engine is not None
