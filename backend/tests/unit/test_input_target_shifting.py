"""
Explicit unit test verifying causal next-token input/target shifting.
"""

import pytest
import json
from pathlib import Path

from app.ml.training import TokenizedDataset


def test_explicit_input_target_shifting(tmp_path):
    """
    EXPLICIT REQUIREMENT:
    Verify tokens = [10, 20, 30, 40, 50] produces:
    input_ids  = [10, 20, 30, 40]
    target_ids = [20, 30, 40, 50]
    """
    jsonl_path = tmp_path / "shifting_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"input_ids": [10, 20, 30, 40, 50]}) + "\n")

    dataset = TokenizedDataset(
        jsonl_path=jsonl_path,
        max_seq_len=4,
        pad_token_id=0,
        vocab_size=100,
    )

    assert len(dataset) == 1
    input_ids, target_ids = dataset[0]

    assert input_ids.tolist() == [10, 20, 30, 40]
    assert target_ids.tolist() == [20, 30, 40, 50]
