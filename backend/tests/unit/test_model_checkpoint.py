"""
Unit tests for model checkpoint saving, loading, round-trip equivalence, and tokenizer version checks.
"""

import pytest
import os
import torch

from app.ml.model import (
    ModelConfig,
    PreciousTransformer,
    save_checkpoint,
    load_checkpoint,
    TokenizerIncompatibilityError,
)


def test_checkpoint_save_load_roundtrip(tmp_path):
    checkpoint_dir = str(tmp_path / "model_v1")

    config = ModelConfig(vocab_size=175, d_model=64, n_heads=2, n_layers=2)
    model_orig = PreciousTransformer(config)
    model_orig.eval()

    save_checkpoint(
        model=model_orig,
        save_dir=checkpoint_dir,
        tokenizer_version="1.0.0",
        tokenizer_hash="test_hash_123",
    )

    # Load checkpoint
    model_loaded, config_loaded, manifest = load_checkpoint(
        save_dir=checkpoint_dir,
        expected_tokenizer_version="1.0.0",
    )

    assert config_loaded.vocab_size == 175
    assert manifest["tokenizer_version"] == "1.0.0"

    # Forward pass equivalence check
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out_orig = model_orig(input_ids)
        out_loaded = model_loaded(input_ids)

    assert torch.allclose(out_orig, out_loaded, atol=1e-5)


def test_checkpoint_tokenizer_incompatibility(tmp_path):
    checkpoint_dir = str(tmp_path / "model_v2")

    config = ModelConfig(vocab_size=175, d_model=64, n_heads=2, n_layers=1)
    model = PreciousTransformer(config)

    save_checkpoint(model=model, save_dir=checkpoint_dir, tokenizer_version="1.0.0")

    with pytest.raises(TokenizerIncompatibilityError):
        load_checkpoint(checkpoint_dir, expected_tokenizer_version="2.0.0")
