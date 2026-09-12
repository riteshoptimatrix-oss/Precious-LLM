"""
Precious Edu LLM — Autoregressive Generation Engine

Implements greedy decoding and generation context sliding window logic for PreciousTransformer.
"""

import torch
from typing import List, Optional, Union

from app.ml.model.transformer import PreciousTransformer
from app.ml.model.exceptions import ModelError


class GreedyGenerator:
    """
    Model-level autoregressive text generator using greedy decoding (argmax).
    """

    def __init__(self, model: PreciousTransformer):
        self.model = model
        self.max_seq_length = model.config.max_seq_length

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Union[List[int], torch.Tensor],
        max_new_tokens: int = 50,
        stop_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively from a prompt.

        Args:
            prompt_ids: List of integer token IDs or 1D/2D PyTorch Tensor.
            max_new_tokens: Maximum number of new tokens to generate (1 <= max_new_tokens <= 2048).
            stop_token_ids: Optional list of token IDs that trigger termination (defaults to [<eos>]).

        Returns:
            Tensor of shape [1, initial_len + generated_len] containing sequence token IDs.
        """
        if max_new_tokens < 1 or max_new_tokens > 2048:
            raise ValueError(f"max_new_tokens must be between 1 and 2048, got {max_new_tokens}")

        # Convert prompt_ids to [1, T] Tensor
        if isinstance(prompt_ids, list):
            if len(prompt_ids) == 0:
                raise ValueError("prompt_ids cannot be empty")
            current_ids = torch.tensor([prompt_ids], dtype=torch.long)
        elif isinstance(prompt_ids, torch.Tensor):
            if prompt_ids.dim() == 1:
                current_ids = prompt_ids.unsqueeze(0)
            elif prompt_ids.dim() == 2:
                current_ids = prompt_ids
            else:
                raise ValueError(f"prompt_ids Tensor must be 1D or 2D, got shape {prompt_ids.shape}")
        else:
            raise TypeError("prompt_ids must be a List[int] or torch.Tensor")

        if stop_token_ids is None:
            stop_token_ids = [self.model.config.eos_token_id]

        device = next(self.model.parameters()).device
        current_ids = current_ids.to(device)

        self.model.eval()

        for _ in range(max_new_tokens):
            # If current sequence exceeds context length, slice to last max_seq_length tokens
            if current_ids.size(1) > self.max_seq_length:
                model_input = current_ids[:, -self.max_seq_length:]
            else:
                model_input = current_ids

            # Forward pass to get logits for last position
            logits = self.model(model_input)  # [1, T_curr, vocab_size]
            next_token_logits = logits[:, -1, :]  # [1, vocab_size]

            # Greedy selection (argmax)
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [1, 1]

            # Append new token to sequence
            current_ids = torch.cat([current_ids, next_token_id], dim=1)

            # Check stop condition
            if next_token_id.item() in stop_token_ids:
                break

        return current_ids
