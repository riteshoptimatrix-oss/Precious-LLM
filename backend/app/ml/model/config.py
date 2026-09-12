"""
Precious Edu LLM — Model Configuration

Externalized model hyperparameters as a dataclass.
Loaded programmatically, from YAML, or directly from tokenizer manifests.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
import os
import yaml

from app.ml.model.exceptions import ModelConfigError


@dataclass
class ModelConfig:
    """
    Configuration for the decoder-only Transformer language model.

    All model dimensions, vocabulary bounds, and architecture choices are defined here.
    """

    # Vocabulary & Special Tokens
    vocab_size: int = 175               # BPE vocabulary size (default updated from Phase 5 tokenizer)
    pad_token_id: int = 0               # <pad> token ID
    bos_token_id: int = 2               # <bos> token ID
    eos_token_id: int = 3               # <eos> token ID
    unk_token_id: int = 1               # <unk> token ID

    # Sequence bounds
    max_seq_length: int = 512           # Maximum context window (tokens)

    # Architectural dimensions
    d_model: int = 256                  # Embedding & hidden dimension
    n_heads: int = 8                    # Number of attention heads
    n_layers: int = 6                   # Number of Transformer blocks
    d_ff: int = 1024                    # Feed-forward inner dimension (typically 4 × d_model)

    # Regularization
    dropout: float = 0.1               # Residual & embedding dropout
    attention_dropout: float = 0.1     # Attention weight dropout

    # Feature flags
    activation: str = "gelu"           # Activation: "gelu" or "relu"
    layer_norm_eps: float = 1e-5       # LayerNorm epsilon
    weight_tying: bool = True          # Tie token embedding and LM head weights
    pre_norm: bool = True              # Pre-LN architecture
    positional_encoding: str = "learned" # "learned" or "sinusoidal"

    # Versioning
    model_version: str = "1.0.0"
    tokenizer_version: str = "1.0.0"

    @property
    def d_k(self) -> int:
        """Dimension of each individual attention head."""
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ModelConfigError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads

    @property
    def estimated_parameters(self) -> int:
        """Calculate total number of model parameters programmatically."""
        token_embed = self.vocab_size * self.d_model
        pos_embed = self.max_seq_length * self.d_model
        
        # Per block: Q, K, V, O projections + FFN up/down + 2 LayerNorms
        attn_params = 4 * self.d_model * self.d_model
        ffn_params = 2 * self.d_model * self.d_ff
        ln_params = 4 * self.d_model  # 2 LayerNorms (weight + bias)
        block_params = attn_params + ffn_params + ln_params
        
        total_blocks = self.n_layers * block_params
        final_ln = 2 * self.d_model
        lm_head = 0 if self.weight_tying else self.d_model * self.vocab_size
        
        return token_embed + pos_embed + total_blocks + final_ln + lm_head

    def validate(self) -> None:
        """Strictly validate configuration parameters."""
        if self.vocab_size <= 0:
            raise ModelConfigError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.max_seq_length <= 0:
            raise ModelConfigError(f"max_seq_length must be > 0, got {self.max_seq_length}")
        if self.d_model <= 0:
            raise ModelConfigError(f"d_model must be > 0, got {self.d_model}")
        if self.n_heads <= 0:
            raise ModelConfigError(f"n_heads must be > 0, got {self.n_heads}")
        if self.n_layers <= 0:
            raise ModelConfigError(f"n_layers must be > 0, got {self.n_layers}")
        if self.d_ff <= 0:
            raise ModelConfigError(f"d_ff must be > 0, got {self.d_ff}")
        if not (0.0 <= self.dropout < 1.0):
            raise ModelConfigError(f"dropout must be in [0, 1), got {self.dropout}")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ModelConfigError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")
        if self.d_model % self.n_heads != 0:
            raise ModelConfigError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.activation not in ("gelu", "relu"):
            raise ModelConfigError(f"activation must be 'gelu' or 'relu', got '{self.activation}'")
        if self.positional_encoding not in ("learned", "sinusoidal"):
            raise ModelConfigError(f"positional_encoding must be 'learned' or 'sinusoidal', got '{self.positional_encoding}'")
        if not (0 <= self.pad_token_id < self.vocab_size):
            raise ModelConfigError(f"pad_token_id ({self.pad_token_id}) out of vocab range (0-{self.vocab_size-1})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Construct configuration from dictionary and validate."""
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load configuration from YAML file."""
        if not os.path.exists(path):
            raise ModelConfigError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "model" in data:
            data = data["model"]
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"model": self.to_dict()}, f, default_flow_style=False, sort_keys=False)
