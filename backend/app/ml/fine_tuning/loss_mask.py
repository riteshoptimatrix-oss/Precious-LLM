"""
Precious Edu LLM — Assistant Loss Mask Builder

Constructs token sequences and target loss masks so that loss is computed
strictly on assistant response tokens, ignoring system prompts, user queries,
role marker tokens, and padding tokens.
"""

from typing import Dict, List, Tuple, Optional
import torch

from app.tokenizer.tokenizer import Tokenizer
from app.tokenizer.config import TokenizerConfig, get_tokenizer_config


class AssistantLossMaskBuilder:
    """
    Builds input tensors and target label masks with assistant-only loss activation (-100 for ignored).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: Optional[TokenizerConfig] = None,
        ignore_index: int = -100,
        max_sequence_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.config = config or get_tokenizer_config()
        self.ignore_index = ignore_index
        self.max_sequence_length = max_sequence_length

        self.pad_id = self.config.special_tokens_map[self.config.PAD_TOKEN]
        self.eos_id = self.config.special_tokens_map[self.config.EOS_TOKEN]
        self.system_id = self.config.special_tokens_map[self.config.SYSTEM_TOKEN]
        self.user_id = self.config.special_tokens_map[self.config.USER_TOKEN]
        self.assistant_id = self.config.special_tokens_map[self.config.ASSISTANT_TOKEN]

    def build_example(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds (input_ids, labels, attention_mask) tensors for a single conversation.

        Returns:
            input_ids: LongTensor [max_sequence_length]
            labels: LongTensor [max_sequence_length] (with ignore_index for non-assistant tokens)
            attention_mask: LongTensor [max_sequence_length] (1 for real tokens, 0 for pad)
        """
        token_ids: List[int] = []
        token_roles: List[str] = []

        for msg in messages:
            role = msg["role"].lower().strip()
            content = msg["content"].strip()

            content_ids = self.tokenizer.encode(content)

            if role == "system":
                token_ids.append(self.system_id)
                token_roles.append("system_header")
                token_ids.extend(content_ids)
                token_roles.extend(["system_content"] * len(content_ids))

            elif role == "user":
                token_ids.append(self.user_id)
                token_roles.append("user_header")
                token_ids.extend(content_ids)
                token_roles.extend(["user_content"] * len(content_ids))

            elif role == "assistant":
                token_ids.append(self.assistant_id)
                token_roles.append("assistant_header")
                token_ids.extend(content_ids)
                token_roles.extend(["assistant_content"] * len(content_ids))
                # append EOS after assistant turn
                token_ids.append(self.eos_id)
                token_roles.append("assistant_content")

        if not token_ids:
            # Fallback for empty messages
            token_ids = [self.pad_id]
            token_roles = ["pad"]

        # Truncate if total sequence exceeds max_sequence_length
        if len(token_ids) > self.max_sequence_length:
            token_ids = token_ids[: self.max_sequence_length]
            token_roles = token_roles[: self.max_sequence_length]

        # Prepare causal shifted inputs and targets
        # Input sequence: tokens[:-1]
        # Target sequence: tokens[1:]
        if len(token_ids) > 1:
            inputs = token_ids[:-1]
            target_ids = token_ids[1:]
            target_roles = token_roles[1:]
        else:
            inputs = token_ids
            target_ids = token_ids
            target_roles = token_roles

        labels: List[int] = []
        for tid, rrole in zip(target_ids, target_roles):
            if rrole == "assistant_content":
                labels.append(tid)
            else:
                labels.append(self.ignore_index)

        attn_mask = [1] * len(inputs)

        # Pad to max_sequence_length - 1 (since sequence length after shifting is L-1)
        target_len = self.max_sequence_length - 1
        pad_amount = target_len - len(inputs)

        if pad_amount > 0:
            inputs.extend([self.pad_id] * pad_amount)
            labels.extend([self.ignore_index] * pad_amount)
            attn_mask.extend([0] * pad_amount)
        elif pad_amount < 0:
            inputs = inputs[:target_len]
            labels = labels[:target_len]
            attn_mask = attn_mask[:target_len]

        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attn_mask, dtype=torch.long),
        )
