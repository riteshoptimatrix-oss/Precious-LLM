"""
Unit tests for FineTuningEvaluator in app.ml.fine_tuning.evaluator
"""

import pytest
import torch
from pathlib import Path

from app.ml.fine_tuning.config import FineTuningConfig
from app.ml.fine_tuning.evaluator import FineTuningEvaluator
from app.ml.model.config import ModelConfig
from app.ml.model.transformer import PreciousTransformer
from app.tokenizer.tokenizer import Tokenizer


def resolve_artifact(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    p2 = Path("..") / path_str
    if p2.exists():
        return p2
    return p


@pytest.fixture
def tokenizer():
    tok_dir = resolve_artifact("artifacts/tokenizer/v1")
    return Tokenizer.load(tok_dir)


@pytest.fixture
def model(tokenizer):
    ckpt_path = resolve_artifact("artifacts/fine_tuning/run-finetune-20260912-132935/final/model.pt")
    if not ckpt_path.exists():
        ckpt_path = resolve_artifact("artifacts/training/run-train_model/checkpoints/latest.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_cfg = ckpt.get("model_config", {})
    cfg = ModelConfig.from_dict(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    m = PreciousTransformer(cfg)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return m


def test_evaluator_golden_prompts(tokenizer, model):
    config = FineTuningConfig(max_sequence_length=10)
    device = torch.device("cpu")

    evaluator = FineTuningEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    results = evaluator.evaluate_behavior(deterministic=True)
    assert len(results) > 0
    for r in results:
        assert "category" in r
        assert "generated_response" in r
        assert "is_relevant" in r
        assert "is_terminated" in r


def test_evaluator_regression(tokenizer, model):
    config = FineTuningConfig()
    device = torch.device("cpu")

    evaluator = FineTuningEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )
