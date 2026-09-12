"""
Precious Edu LLM — Tokenizer Decoding Unit Tests
"""

import pytest
from app.tokenizer.config import get_tokenizer_config
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.encoder import BPEEncoder
from app.tokenizer.decoder import BPEDecoder


@pytest.fixture
def sample_tokenizer_components():
    config = get_tokenizer_config()
    vocab = Vocabulary()
    for st, st_id in config.special_tokens_map.items():
        vocab.add_token(st, forced_id=st_id)

    for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;-_()[]{}/\\'\"@#$%&*+= ":
        vocab.add_token(ch)

    merges = [("H", "e"), ("l", "l"), ("o", " "), ("i", "s")]
    vocab.add_token("He")
    vocab.add_token("ll")
    vocab.add_token("o ")
    vocab.add_token("is")

    encoder = BPEEncoder(vocab, merges, config)
    decoder = BPEDecoder(vocab, config)
    return encoder, decoder


def test_roundtrip_decoding(sample_tokenizer_components):
    encoder, decoder = sample_tokenizer_components
    original_text = "Hello, my name is Ritesh."

    encoded_ids = encoder.encode(original_text)
    decoded_text = decoder.decode(encoded_ids)

    assert decoded_text == original_text


def test_decode_skip_special_tokens(sample_tokenizer_components):
    encoder, decoder = sample_tokenizer_components
    original_text = "Hello world"

    encoded_ids = encoder.encode(original_text, add_bos=True, add_eos=True)

    decoded_with_specials = decoder.decode(encoded_ids, skip_special_tokens=False)
    decoded_clean = decoder.decode(encoded_ids, skip_special_tokens=True)

    assert "<bos>" in decoded_with_specials
    assert "<eos>" in decoded_with_specials
    assert decoded_clean == original_text
