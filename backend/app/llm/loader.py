"""
Precious Edu LLM — Model Loader

Dedicated loader for initializing Phase 8 fine-tuned Transformer model & Phase 5 tokenizer.
Performs strict compatibility checks between checkpoint metadata and tokenizer vocabulary.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import torch

from app.llm.config import LLMConfig
from app.ml.fine_tuning.exceptions import CheckpointCompatibilityError
from app.ml.model.config import ModelConfig
from app.ml.model.transformer import PreciousTransformer
from app.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class CustomLLMModelLoader:
    """
    Loader responsible for constructing, validating, and moving model & tokenizer into memory.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def resolve_device(self) -> torch.device:
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.config.device == "cpu":
            return torch.device("cpu")
        elif self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def load(self) -> Tuple[PreciousTransformer, Tokenizer, ModelConfig, Path]:
        """
        Loads Phase 5 tokenizer and Phase 8 fine-tuned model checkpoint.

        Returns:
            (model, tokenizer, model_config, checkpoint_path)
        """
        # 1. Resolve & Load Tokenizer
        tokenizer_path = self.config.resolve_tokenizer_dir()
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = Tokenizer.load(tokenizer_path)

        # 2. Resolve & Load Checkpoint
        ckpt_path = self.config.resolve_checkpoint_path()
        logger.info(f"Loading LLM model checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # 3. Extract & Validate ModelConfig
        raw_config = checkpoint.get("model_config", {})
        if isinstance(raw_config, dict):
            model_config = ModelConfig.from_dict(raw_config)
        else:
            model_config = raw_config

        # 4. Strict Compatibility Verification
        if model_config.vocab_size != tokenizer.vocab_size:
            raise CheckpointCompatibilityError(
                f"Vocabulary size mismatch: Model config has {model_config.vocab_size}, "
                f"but Tokenizer has {tokenizer.vocab_size}."
            )

        # 5. Instantiate Model & Load State Dict
        model = PreciousTransformer(model_config)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)

        # 6. Device Placement & Eval Mode
        device = self.resolve_device()
        model.to(device)
        model.eval()

        logger.info(
            f"Successfully loaded Precious AI LLM ({model.num_parameters['total']:,} params) "
            f"on device '{device}'."
        )

        return model, tokenizer, model_config, ckpt_path
