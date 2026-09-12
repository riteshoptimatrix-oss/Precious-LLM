"""
Precious Edu LLM — Embedding Layers

Implements token embeddings, positional embeddings, and combined embedding module.
"""

import torch
import torch.nn as nn

from app.ml.model.config import ModelConfig
from app.ml.model.exceptions import ContextLengthExceededError


class TokenEmbedding(nn.Module):
    """Maps discrete token IDs to d_model dimensional vectors."""

    def __init__(self, vocab_size: int, d_model: int, pad_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [B, T] with integer token IDs.

        Returns:
            Tensor of shape [B, T, d_model].
        """
        return self.embedding(input_ids)


class LearnedPositionalEmbedding(nn.Module):
    """Maps sequence position indices [0..T-1] to d_model dimensional vectors."""

    def __init__(self, max_seq_length: int, d_model: int):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.embedding = nn.Embedding(max_seq_length, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Tensor of shape [1, T] or [B, T] with position indices.

        Returns:
            Tensor of shape [1, T, d_model] or [B, T, d_model].
        """
        return self.embedding(positions)


class CombinedEmbedding(nn.Module):
    """
    Combines TokenEmbedding and LearnedPositionalEmbedding with dropout.
    Enforces context_length constraints.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.max_seq_length = config.max_seq_length
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            pad_token_id=config.pad_token_id,
        )
        self.positional_embedding = LearnedPositionalEmbedding(
            max_seq_length=config.max_seq_length,
            d_model=config.d_model,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [B, T] with integer token IDs.

        Returns:
            Tensor of shape [B, T, d_model].
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_length:
            raise ContextLengthExceededError(
                f"Input sequence length ({seq_len}) exceeds configured maximum "
                f"context window length ({self.max_seq_length})."
            )

        # Generate position IDs [0, 1, ..., seq_len - 1] on same device
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)  # [1, T]

        tok_emb = self.token_embedding(input_ids)          # [B, T, d_model]
        pos_emb = self.positional_embedding(positions)    # [1, T, d_model]

        x = tok_emb + pos_emb                             # Broadcasts over B: [B, T, d_model]
        return self.dropout(x)
