"""
Unit tests for TokenizedDataset processing, chunking, padding, and bounds validation.
"""

import pytest
import json
from pathlib import Path

from app.ml.training import TokenizedDataset
from app.ml.training.exceptions import DataLoadingError


def test_dataset_loading_and_shifting(tmp_path):
    jsonl_path = tmp_path / "train_tokenized.jsonl"
    data = [
        {"input_ids": [2, 10, 20, 30, 40, 50, 3], "num_tokens": 7},
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    dataset = TokenizedDataset(
        jsonl_path=jsonl_path,
        max_seq_len=6,
        pad_token_id=0,
        vocab_size=100,
    )

    assert len(dataset) == 1
    input_ids, target_ids = dataset[0]

    assert input_ids.tolist() == [2, 10, 20, 30, 40, 50]
    assert target_ids.tolist() == [10, 20, 30, 40, 50, 3]


def test_dataset_padding_short_sequence(tmp_path):
    jsonl_path = tmp_path / "short_tokenized.jsonl"
    data = [{"input_ids": [2, 10, 3], "num_tokens": 3}]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    dataset = TokenizedDataset(
        jsonl_path=jsonl_path,
        max_seq_len=5,
        pad_token_id=0,
        vocab_size=100,
    )

    assert len(dataset) == 1
    input_ids, target_ids = dataset[0]

    # Target sequence length target_token_count = 6 (padded to 6, inputs length 5)
    assert input_ids.tolist() == [2, 10, 3, 0, 0]
    assert target_ids.tolist() == [10, 3, 0, 0, 0]


def test_dataset_out_of_bounds_token_id(tmp_path):
    jsonl_path = tmp_path / "corrupt_tokenized.jsonl"
    data = [{"input_ids": [2, 999, 3], "num_tokens": 3}]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    with pytest.raises(DataLoadingError):
        _ = TokenizedDataset(jsonl_path=jsonl_path, vocab_size=100)
