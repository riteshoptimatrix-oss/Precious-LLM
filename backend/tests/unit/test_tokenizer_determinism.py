"""
Precious Edu LLM — Tokenizer Determinism Unit Tests
"""

import pytest
from app.tokenizer.config import get_tokenizer_config
from app.tokenizer.trainer import BPETrainer


def test_bpe_training_determinism():
    config = get_tokenizer_config()
    sequences = [
        "Hello, my name is Ritesh.",
        "Hello world!",
        "Thanks for your help.",
        "Goodbye and see you later!"
    ]

    trainer1 = BPETrainer(config=config)
    vocab1, merges1 = trainer1.train_from_sequences(sequences)

    trainer2 = BPETrainer(config=config)
    vocab2, merges2 = trainer2.train_from_sequences(sequences)

    assert vocab1.token_to_id == vocab2.token_to_id
    assert merges1 == merges2
