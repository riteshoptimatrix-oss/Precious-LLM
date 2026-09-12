"""
Precious Edu LLM — Pretraining Evaluator

Evaluates validation and test datasets without gradient computation.
Calculates mean cross-entropy loss and safe perplexity.
Restores training state upon completion.
"""

import math
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader

from app.ml.model import PreciousTransformer


class Evaluator:
    """
    Validation and test evaluation engine.
    """

    def __init__(
        self,
        model: PreciousTransformer,
        data_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> Tuple[float, float]:
        """
        Run evaluation loop over dataset.

        Args:
            max_batches: Optional limit on number of evaluation batches.

        Returns:
            Tuple of (mean_loss, perplexity).
        """
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        for idx, (input_ids, target_ids) in enumerate(self.data_loader):
            if max_batches is not None and idx >= max_batches:
                break

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            logits, loss = self.model(input_ids, targets=target_ids)

            total_loss += loss.item()
            batch_count += 1

        if was_training:
            self.model.train()

        if batch_count == 0:
            return 0.0, 1.0

        mean_loss = total_loss / batch_count
        
        # Calculate perplexity safely (cap at 1e8 to avoid OverflowError)
        try:
            perplexity = math.exp(mean_loss)
            if math.isinf(perplexity) or perplexity > 1e8:
                perplexity = 1e8
        except OverflowError:
            perplexity = 1e8

        return mean_loss, perplexity
