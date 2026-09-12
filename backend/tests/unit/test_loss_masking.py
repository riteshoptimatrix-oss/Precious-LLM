"""
Unit tests for AssistantLossMaskBuilder in app.ml.fine_tuning.loss_mask
"""

import pytest
import torch
from pathlib import Path

from app.ml.fine_tuning.loss_mask import AssistantLossMaskBuilder
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


def test_assistant_loss_masking_behavior(tokenizer):
    builder = AssistantLossMaskBuilder(
        tokenizer=tokenizer,
        ignore_index=-100,
        max_sequence_length=64
    )

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    input_ids, labels, attn_mask = builder.build_example(messages)

    assert input_ids.shape[0] == 63  # shifted length (64 - 1)
    assert labels.shape[0] == 63
    assert attn_mask.shape[0] == 63

    # Check that labels contain active targets (not all -100)
    active_labels = [lbl.item() for lbl in labels if lbl.item() != -100]
    assert len(active_labels) > 0
