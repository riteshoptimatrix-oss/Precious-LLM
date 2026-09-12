"""
Precious Edu LLM — PreciousTransformer Language Model

Complete decoder-only Transformer language model implemented from scratch.
Composes CombinedEmbedding, Multi-Head Attention, TransformerBlocks, Final LayerNorm,
and LM Head with optional Weight Tying.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict

from app.ml.model.config import ModelConfig
from app.ml.model.embeddings import CombinedEmbedding
from app.ml.model.layers import TransformerBlock
from app.ml.model.initialization import init_weights


class PreciousTransformer(nn.Module):
    """
    Decoder-only GPT-style Transformer Language Model.

    Input: Token IDs [B, T]
    Output: Logits [B, T, vocab_size] and optional cross-entropy loss.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        # 1. Embeddings (Token + Position + Dropout)
        self.embedding = CombinedEmbedding(config)

        # 2. Stack of N Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # 3. Final Layer Normalization
        self.final_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # 4. Language Model Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 5. Weight Tying (Token Embedding <-> LM Head Output Projection)
        if config.weight_tying:
            self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # 6. Initialize parameters
        self.apply(init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Tensor of shape [B, T] with integer token IDs.
            targets: Optional Tensor of shape [B, T] with target token IDs for loss computation.
            attention_mask: Optional additional mask tensor.

        Returns:
            If targets is None: logits [B, T, vocab_size]
            If targets is provided: tuple of (logits, loss)
        """
        # Embed tokens & positions
        x = self.embedding(input_ids)  # [B, T, d_model]

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # Final Layer Normalization
        x = self.final_ln(x)

        # Map hidden representations to vocabulary logits
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # Compute next-token prediction cross-entropy loss if targets provided
        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross entropy calculation
            logits_flat = logits.view(-1, self.config.vocab_size)
            targets_flat = targets.view(-1)
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fn(logits_flat, targets_flat)
            return logits, loss

        return logits

    @property
    def num_parameters(self) -> Dict[str, int]:
        """Calculate exact total, trainable, and non-trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
        }
