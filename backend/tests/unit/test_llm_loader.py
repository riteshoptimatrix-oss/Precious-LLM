"""
Unit tests for CustomLLMModelLoader in app.llm.loader
"""

import pytest
from pathlib import Path
from app.llm.config import LLMConfig
from app.llm.loader import CustomLLMModelLoader


def test_llm_loader_initialization():
    config = LLMConfig()
    loader = CustomLLMModelLoader(config)

    model, tokenizer, model_config, ckpt_path = loader.load()

    assert model is not None
    assert tokenizer is not None
    assert model_config is not None
    assert ckpt_path.exists()
    assert model_config.vocab_size == tokenizer.vocab_size
