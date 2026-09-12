"""
Precious Edu LLM — Tokenizer Serialization Unit Tests
"""

import pytest
from pathlib import Path
from app.tokenizer.config import get_tokenizer_config
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.tokenizer import Tokenizer


def test_serialization_save_and_load(tmp_path: Path):
    config = get_tokenizer_config()
    vocab = Vocabulary()
    for st, st_id in config.special_tokens_map.items():
        vocab.add_token(st, forced_id=st_id)

    for ch in "abcdefghijklmnopqrstuvwxyz ":
        vocab.add_token(ch)

    merges = [("h", "e"), ("l", "l")]
    vocab.add_token("he")
    vocab.add_token("ll")

    original_tok = Tokenizer(vocab=vocab, merges=merges, config=config)
    save_dir = tmp_path / "tokenizer_v1"
    original_tok.save(save_dir)

    loaded_tok = Tokenizer.load(save_dir, config=config)

    test_text = "hello"
    assert loaded_tok.encode(test_text) == original_tok.encode(test_text)
    assert loaded_tok.decode(loaded_tok.encode(test_text)) == original_tok.decode(original_tok.encode(test_text))
