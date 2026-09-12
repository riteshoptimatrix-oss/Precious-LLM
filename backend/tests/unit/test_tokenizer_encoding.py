"""
Precious Edu LLM — Tokenizer Encoding Unit Tests
"""

import pytest
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.encoder import BPEEncoder
from app.tokenizer.config import get_tokenizer_config


@pytest.fixture
def sample_encoder():
    config = get_tokenizer_config()
    vocab = Vocabulary()
    for st, st_id in config.special_tokens_map.items():
        vocab.add_token(st, forced_id=st_id)

    for ch in "abcdefghijklmnopqrstuvwxyz0123456789.,!?:;-_()[]{}/\\'\"@#$%&*+= ":
        vocab.add_token(ch)

    merges = [("h", "e"), ("l", "l"), ("o", " "), ("i", "s")]
    vocab.add_token("he")
    vocab.add_token("ll")
    vocab.add_token("o ")
    vocab.add_token("is")

    return BPEEncoder(vocab, merges, config)


def test_encode_basic_text(sample_encoder):
    ids = sample_encoder.encode("hello world")
    assert len(ids) > 0
    assert all(isinstance(i, int) and i >= 0 for i in ids)


def test_encode_with_bos_eos(sample_encoder):
    ids = sample_encoder.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == sample_encoder.config.special_tokens_map["<bos>"]
    assert ids[-1] == sample_encoder.config.special_tokens_map["<eos>"]


def test_encode_empty_text(sample_encoder):
    ids = sample_encoder.encode("")
    assert ids == []

    ids_bos_eos = sample_encoder.encode("", add_bos=True, add_eos=True)
    assert ids_bos_eos == [
        sample_encoder.config.special_tokens_map["<bos>"],
        sample_encoder.config.special_tokens_map["<eos>"]
    ]


def test_encode_whitespace_preservation(sample_encoder):
    ids1 = sample_encoder.encode("Hello")
    ids2 = sample_encoder.encode(" Hello")
    ids3 = sample_encoder.encode("Hello ")

    assert ids1 != ids2
    assert ids1 != ids3
