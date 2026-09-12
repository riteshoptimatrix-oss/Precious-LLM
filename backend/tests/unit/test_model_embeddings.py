"""
Unit tests for PreciousTransformer embedding modules.
"""

import pytest
import torch

from app.ml.model import ModelConfig, TokenEmbedding, LearnedPositionalEmbedding, CombinedEmbedding
from app.ml.model.exceptions import ContextLengthExceededError


def test_token_embedding():
    vocab_size = 175
    d_model = 256
    embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, pad_token_id=0)

    input_ids = torch.tensor([[0, 5, 10, 100]], dtype=torch.long)
    output = embed(input_ids)

    assert output.shape == (1, 4, d_model)
    assert not torch.isnan(output).any()


def test_positional_embedding():
    max_seq_length = 512
    d_model = 256
    pos_embed = LearnedPositionalEmbedding(max_seq_length=max_seq_length, d_model=d_model)

    positions = torch.arange(0, 10).unsqueeze(0)
    output = pos_embed(positions)

    assert output.shape == (1, 10, d_model)
    assert not torch.isnan(output).any()


def test_combined_embedding():
    config = ModelConfig(vocab_size=175, max_seq_length=512, d_model=256)
    combined = CombinedEmbedding(config)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    output = combined(input_ids)

    assert output.shape == (2, 3, 256)
    assert not torch.isnan(output).any()


def test_combined_embedding_context_exceeded():
    config = ModelConfig(vocab_size=175, max_seq_length=10, d_model=64)
    combined = CombinedEmbedding(config)

    input_ids = torch.zeros((1, 15), dtype=torch.long)
    with pytest.raises(ContextLengthExceededError):
        combined(input_ids)
