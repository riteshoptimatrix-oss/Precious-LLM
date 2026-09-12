"""
Precious Edu LLM — Tokenizer Unicode & Multilingual Unit Tests
"""

import pytest
from app.tokenizer.config import get_tokenizer_config
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.encoder import BPEEncoder
from app.tokenizer.decoder import BPEDecoder


def test_unicode_multilingual_roundtrip():
    config = get_tokenizer_config()
    vocab = Vocabulary()
    for st, st_id in config.special_tokens_map.items():
        vocab.add_token(st, forced_id=st_id)

    sample_texts = [
        "Hello World",
        "नमस्ते, मेरा नाम रितेश है।",
        "你好世界",
        "こんにちは",
        "مرحبا",
        "🙂 😀 🚀",
        "₹1000",
        "F1 visa M1 visa USA"
    ]

    # Add all characters to vocabulary
    for text in sample_texts:
        for ch in text:
            if ch not in vocab:
                vocab.add_token(ch)

    encoder = BPEEncoder(vocab, [], config)
    decoder = BPEDecoder(vocab, config)

    for text in sample_texts:
        encoded = encoder.encode(text)
        decoded = decoder.decode(encoded)
        assert decoded == text, f"Failed roundtrip for text: {text}"
