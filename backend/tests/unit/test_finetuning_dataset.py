"""
Unit tests for ConversationalFineTuningDataset in app.ml.fine_tuning.dataset
"""

import pytest
import json
from pathlib import Path

from app.ml.fine_tuning.config import FineTuningConfig
from app.ml.fine_tuning.dataset import ConversationalFineTuningDataset
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


def test_conversational_dataset_loading(tmp_path, tokenizer):
    data = [
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]},
        {"instruction": "What is F1?", "response": "Student visa."},
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}, # duplicate
        {"invalid": "format"} # invalid
    ]

    jsonl_file = tmp_path / "test_data.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    config = FineTuningConfig(max_sequence_length=32)
    dataset = ConversationalFineTuningDataset(
        jsonl_path=str(jsonl_file),
        tokenizer=tokenizer,
        config=config,
    )

    assert len(dataset) == 2  # 2 valid unique examples
