"""
Precious Edu LLM — Text Generator

Autoregressive token-by-token text generator supporting greedy decoding, temperature scaling,
top-K sampling, top-P (nucleus) sampling, repetition penalty, and custom stop tokens.
"""

import logging
from typing import List, Optional, Union
import torch

from app.ml.fine_tuning.exceptions import GenerationError
from app.tokenizer.config import get_tokenizer_config

logger = logging.getLogger(__name__)


class TextGenerator:
    """
    Autoregressive text generation engine for PreciousTransformer models.
    """

    def __init__(self, model: torch.nn.Module, eos_token_id: int = 3, pad_token_id: int = 0):
        self.model = model
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Generate new token IDs autoregressively given an input prompt tensor.

        Args:
            input_ids: LongTensor [B, T] or [T] prompt token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 or top_k=1 for deterministic greedy decoding)
            top_k: Top-K filter threshold
            top_p: Top-P (nucleus) cumulative probability threshold
            repetition_penalty: Multiplicative penalty for previously seen tokens
            stop_tokens: Additional token IDs that trigger early stopping

        Returns:
            LongTensor [B, T + generated_length] full output sequence
        """
        self.model.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        device = input_ids.device
        batch_size, prompt_len = input_ids.shape

        stop_token_set = set(stop_tokens or [])
        stop_token_set.add(self.eos_token_id)

        context_len = getattr(self.model.config, "max_seq_length", getattr(self.model.config, "context_length", 256))
        if prompt_len > context_len:
            input_ids = input_ids[:, -context_len:]

        curr_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            if curr_ids.shape[1] > context_len:
                model_inputs = curr_ids[:, -context_len:]
            else:
                model_inputs = curr_ids

            # Forward pass to get logits for last position
            logits = self.model(model_inputs)  # [B, T_curr, vocab_size]
            next_token_logits = logits[:, -1, :].clone()  # [B, vocab_size]

            # Repetition Penalty
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    seen = set(curr_ids[b].tolist())
                    for token_id in seen:
                        if next_token_logits[b, token_id] < 0:
                            next_token_logits[b, token_id] *= repetition_penalty
                        else:
                            next_token_logits[b, token_id] /= repetition_penalty

            # Greedy Decoding
            if temperature == 0.0 or top_k == 1:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Temperature scaling
                next_token_logits = next_token_logits / temperature

                # Top-K Filtering
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

                # Top-P (Nucleus) Filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above top_p threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift mask right to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    for b in range(batch_size):
                        indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                        next_token_logits[b, indices_to_remove] = -float("Inf")

                # Sample from softmax probability distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

            curr_ids = torch.cat([curr_ids, next_tokens], dim=1)

            # Check stopping condition (if single sequence or all batch elements produced stop token)
            produced_token = next_tokens[0, 0].item()
            if produced_token in stop_token_set:
                break

        return curr_ids
