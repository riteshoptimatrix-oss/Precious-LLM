"""
Precious Edu LLM — Model Package

Exports model configuration, architecture components, checkpoints, and generator.
"""

from app.ml.model.config import ModelConfig
from app.ml.model.exceptions import (
    ModelError,
    ModelConfigError,
    CheckpointError,
    ContextLengthExceededError,
    TokenizerIncompatibilityError,
)
from app.ml.model.masking import create_causal_mask
from app.ml.model.embeddings import TokenEmbedding, LearnedPositionalEmbedding, CombinedEmbedding
from app.ml.model.attention import MultiHeadCausalAttention
from app.ml.model.layers import FeedForward, TransformerBlock
from app.ml.model.initialization import init_weights
from app.ml.model.transformer import PreciousTransformer
from app.ml.model.checkpoint import save_checkpoint, load_checkpoint
from app.ml.model.generation import GreedyGenerator

__all__ = [
    "ModelConfig",
    "ModelError",
    "ModelConfigError",
    "CheckpointError",
    "ContextLengthExceededError",
    "TokenizerIncompatibilityError",
    "create_causal_mask",
    "TokenEmbedding",
    "LearnedPositionalEmbedding",
    "CombinedEmbedding",
    "MultiHeadCausalAttention",
    "FeedForward",
    "TransformerBlock",
    "init_weights",
    "PreciousTransformer",
    "save_checkpoint",
    "load_checkpoint",
    "GreedyGenerator",
]
