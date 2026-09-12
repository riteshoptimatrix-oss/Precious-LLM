"""
Precious Edu LLM — Tokenizer Special Tokens Unit Tests
"""

import pytest
from app.tokenizer.config import get_tokenizer_config
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.encoder import BPEEncoder


def test_special_token_stable_ids():
    config = get_tokenizer_config()
    st_map = config.special_tokens_map

    assert st_map["<pad>"] == 0
    assert st_map["<unk>"] == 1
    assert st_map["<bos>"] == 2
    assert st_map["<eos>"] == 3
    assert st_map["<system>"] == 4
    assert st_map["<user>"] == 5
    assert st_map["<assistant>"] == 6

    # Verify no duplicate IDs
    ids = list(st_map.values())
    assert len(ids) == len(set(ids))


def test_special_tokens_not_split_by_bpe():
    config = get_tokenizer_config()
    vocab = Vocabulary()
    for st, st_id in config.special_tokens_map.items():
        vocab.add_token(st, forced_id=st_id)

    for ch in "userassistanthello":
        vocab.add_token(ch)

    # Even if "u", "s", "e", "r" are in vocabulary, "<user>" string must be matched as ID 5
    encoder = BPEEncoder(vocab, [], config)
    encoded = encoder.encode("<user>\nHello")

    assert encoded[0] == 5  # <user> token ID
