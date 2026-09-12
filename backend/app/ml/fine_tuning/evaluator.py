"""
Precious Edu LLM — Fine-Tuning Evaluator & Behavioral Benchmark

Computes validation loss, perplexity, and performs behavioral evaluations on golden prompts
including baseline vs fine-tuned comparison and catastrophic forgetting checks.
"""

import math
import logging
from typing import Dict, List, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader

from app.ml.fine_tuning.config import FineTuningConfig
from app.ml.fine_tuning.formatter import ConversationFormatter
from app.ml.inference.generator import TextGenerator
from app.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

GOLDEN_PROMPTS = [
    {
        "category": "greeting",
        "prompt": [{"role": "user", "content": "Hello"}],
        "keywords": ["hello", "hi", "assist", "help", "welcome"],
    },
    {
        "category": "name",
        "prompt": [{"role": "user", "content": "My name is Ritesh."}],
        "keywords": ["ritesh", "nice", "meet", "pleasure", "hello"],
    },
    {
        "category": "thanks",
        "prompt": [{"role": "user", "content": "Thank you."}],
        "keywords": ["welcome", "happy", "help", "anytime", "pleasure"],
    },
    {
        "category": "goodbye",
        "prompt": [{"role": "user", "content": "Bye."}],
        "keywords": ["goodbye", "bye", "see", "great", "day"],
    },
    {
        "category": "question",
        "prompt": [{"role": "user", "content": "What is a student visa?"}],
        "keywords": ["visa", "student", "study", "country", "education"],
    },
]

REGRESSION_PROMPTS = [
    "Education is important because",
    "A university student must",
    "To learn new skills",
]


class FineTuningEvaluator:
    """
    Evaluator for loss metrics, golden prompt behavior analysis, and model comparisons.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        config: FineTuningConfig,
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.formatter = ConversationFormatter(config=self.tokenizer.config)
        self.generator = TextGenerator(
            model=model,
            eos_token_id=self.tokenizer.config.special_tokens_map[self.tokenizer.config.EOS_TOKEN],
            pad_token_id=self.tokenizer.config.special_tokens_map[self.tokenizer.config.PAD_TOKEN],
        )

    @torch.no_grad()
    def evaluate_dataset(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate dataset loss and perplexity with assistant-only loss masking.

        Returns:
            (loss, perplexity)
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)

        for input_ids, labels, attention_mask in dataloader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)

            logits = self.model(input_ids, attention_mask=attention_mask)

            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            loss = loss_fn(logits_flat, labels_flat)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                total_batches += 1

        avg_loss = (total_loss / total_batches) if total_batches > 0 else float("nan")
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        return avg_loss, perplexity

    def evaluate_behavior(
        self,
        golden_prompts: Optional[List[Dict[str, Any]]] = None,
        deterministic: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate model on golden prompts for behavioral compliance.
        """
        prompts = golden_prompts or GOLDEN_PROMPTS
        results = []

        temp = 0.0 if deterministic else self.config.temperature
        top_k = 1 if deterministic else self.config.top_k

        for item in prompts:
            prompt_msgs = item["prompt"]
            prompt_str = self.formatter.format_prompt_only(prompt_msgs)
            encoded = self.tokenizer.encode(prompt_str)

            input_tensor = torch.tensor(encoded, dtype=torch.long, device=self.device)

            output_tensor = self.generator.generate(
                input_tensor,
                max_new_tokens=self.config.max_new_tokens,
                temperature=temp,
                top_k=top_k,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
            )

            gen_ids = output_tensor[0, len(encoded):].tolist()
            decoded_response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Behavioral criteria analysis
            lowered = decoded_response.lower()
            keyword_matches = [kw for kw in item.get("keywords", []) if kw in lowered]

            is_relevant = len(keyword_matches) > 0 or len(decoded_response) > 2
            is_terminated = (self.tokenizer.config.special_tokens_map[self.tokenizer.config.EOS_TOKEN] in gen_ids) or (len(gen_ids) < self.config.max_new_tokens)
            has_repetition = len(gen_ids) > 10 and (len(set(gen_ids)) / len(gen_ids)) < 0.3

            results.append({
                "category": item["category"],
                "prompt": prompt_msgs[-1]["content"],
                "formatted_prompt": prompt_str,
                "generated_response": decoded_response,
                "token_count": len(gen_ids),
                "is_relevant": is_relevant,
                "is_terminated": is_terminated,
                "has_repetition": has_repetition,
                "matched_keywords": keyword_matches,
            })

        return results

    def evaluate_regression(self, prompts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Evaluates model on general language prompts to monitor catastrophic forgetting.
        """
        prompts_to_test = prompts or REGRESSION_PROMPTS
        results = []

        for p in prompts_to_test:
            encoded = self.tokenizer.encode(p)
            input_tensor = torch.tensor(encoded, dtype=torch.long, device=self.device)

            output_tensor = self.generator.generate(
                input_tensor,
                max_new_tokens=32,
                temperature=0.0,
                top_k=1,
            )

            gen_ids = output_tensor[0, len(encoded):].tolist()
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            results.append({
                "prompt": p,
                "continuation": decoded,
                "token_count": len(gen_ids),
            })

        return results
