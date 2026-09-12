"""
Unit tests for TextGenerator in app.ml.inference.generator
"""

import pytest
import torch

from app.ml.model.config import ModelConfig
from app.ml.model.transformer import PreciousTransformer
from app.ml.inference.generator import TextGenerator


@pytest.fixture
def model():
    cfg = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=64,
    )
    return PreciousTransformer(cfg)


def test_generator_greedy_decoding(model):
    generator = TextGenerator(model=model, eos_token_id=3, pad_token_id=0)
    input_ids = torch.tensor([[2, 5, 10]], dtype=torch.long)

    output = generator.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        temperature=0.0,
        top_k=1,
    )

    assert output.shape[0] == 1
    assert output.shape[1] <= 3 + 5
    assert output.shape[1] > 3


def test_generator_sampling(model):
    generator = TextGenerator(model=model, eos_token_id=3, pad_token_id=0)
    input_ids = torch.tensor([[2, 5, 10]], dtype=torch.long)

    output = generator.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        temperature=0.7,
        top_k=10,
        top_p=0.9,
    )

    assert output.shape[0] == 1
    assert output.shape[1] > 3
