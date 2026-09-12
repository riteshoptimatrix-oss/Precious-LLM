"""
Precious Edu LLM — Tokenizer Data Leakage Prevention Unit Tests
"""

import pytest
from pathlib import Path
from app.tokenizer.config import get_tokenizer_config
from app.tokenizer.trainer import BPETrainer
from app.tokenizer.exceptions import TokenizerTrainingError


def test_tokenizer_training_rejects_validation_data(tmp_path: Path):
    config = get_tokenizer_config()
    trainer = BPETrainer(config=config)

    val_file = tmp_path / "validation.jsonl"
    val_file.write_text('{"text": "validation sequence"}\n', encoding="utf-8")

    with pytest.raises(TokenizerTrainingError) as exc_info:
        trainer.train_from_file(val_file)

    assert "Data leakage rule violation" in str(exc_info.value)


def test_tokenizer_training_rejects_evaluation_data(tmp_path: Path):
    config = get_tokenizer_config()
    trainer = BPETrainer(config=config)

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_file = eval_dir / "golden_eval.jsonl"
    eval_file.write_text('{"text": "eval sequence"}\n', encoding="utf-8")

    with pytest.raises(TokenizerTrainingError) as exc_info:
        trainer.train_from_file(eval_file)

    assert "Data leakage rule violation" in str(exc_info.value)
