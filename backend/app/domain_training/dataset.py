"""
Precious Edu LLM — Domain Training PyTorch Dataset

Formats DomainRecord items into prompt strings, encodes them with Phase 5 Tokenizer,
and applies assistant-only loss masking (-100 for non-target tokens).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset

from app.domain.schema import DomainExampleType, DomainRecord
from app.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class DomainTrainingDataset(Dataset):
    """
    PyTorch Dataset for Phase 11 domain fine-tuning with assistant-only loss masking.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Tokenizer,
        max_length: int = 64,
        max_sequence_length: Optional[int] = None
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_sequence_length if max_sequence_length is not None else max_length

        self.examples: List[Dict[str, Any]] = []
        self._load_and_process()

    def _load_and_process(self) -> None:
        """Load records from JSONL file and format into tokenized input/label tensors."""
        if not self.data_path.exists():
            logger.warning(f"Domain training dataset path not found: {self.data_path}")
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                rec = DomainRecord(**data)
                prompt_str, response_str = self._format_record(rec)
                if prompt_str and response_str:
                    item = self._tokenize_and_mask(prompt_str, response_str)
                    if item:
                        self.examples.append(item)

    def _format_record(self, rec: DomainRecord) -> Tuple[Optional[str], Optional[str]]:
        """Formats a DomainRecord into (prompt_string, response_string)."""
        st_type = rec.type

        if st_type in (DomainExampleType.INSTRUCTION_RESPONSE, DomainExampleType.CLARIFICATION, DomainExampleType.AMBIGUITY, DomainExampleType.NEGATIVE_EXAMPLE):
            prompt = f"<system>\nYou are Precious AI, an expert consultancy assistant.\n<user>\n{rec.instruction}\n<assistant>\n"
            response = rec.response
            return prompt, response

        elif st_type == DomainExampleType.TERMINOLOGY:
            prompt = f"<system>\nYou are Precious AI.\n<user>\nWhat is {rec.term}?\n<assistant>\n"
            response = rec.definition
            return prompt, response

        elif st_type == DomainExampleType.SERVICE:
            prompt = f"<system>\nYou are Precious AI.\n<user>\nTell me about {rec.service_name}.\n<assistant>\n"
            response = rec.response
            return prompt, response

        elif st_type == DomainExampleType.CONVERSATION:
            if not rec.messages or len(rec.messages) < 2:
                return None, None
            turns = []
            for m in rec.messages[:-1]:
                r = m.get("role")
                c = m.get("content")
                if r == "user":
                    turns.append(f"<user>\n{c}")
                elif r == "assistant":
                    turns.append(f"<assistant>\n{c}")

            prompt = f"<system>\nYou are Precious AI.\n" + "\n".join(turns) + f"\n<assistant>\n"
            response = rec.messages[-1].get("content")
            return prompt, response

        return None, None

    def _tokenize_and_mask(self, prompt_str: str, response_str: str) -> Optional[Dict[str, torch.Tensor]]:
        """Encodes prompt and response, applying loss mask -100 to prompt tokens."""
        prompt_ids = self.tokenizer.encode(prompt_str)
        response_ids = self.tokenizer.encode(response_str)

        full_ids = prompt_ids + response_ids
        eos_id = self.tokenizer.config.special_tokens_map[self.tokenizer.config.EOS_TOKEN]
        full_ids.append(eos_id)

        # Truncate if needed
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        # Target labels: mask prompt tokens with -100
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # Truncate labels to match full_ids
        labels = labels[:len(full_ids)]

        # Padding
        pad_id = self.tokenizer.config.special_tokens_map[self.tokenizer.config.PAD_TOKEN]
        pad_len = self.max_length - len(full_ids)

        if pad_len > 0:
            full_ids = full_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]
