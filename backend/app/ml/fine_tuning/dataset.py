"""
Precious Edu LLM — Conversational Dataset Loader

PyTorch Dataset implementation for loading single-turn, multi-turn, and instruction-response datasets.
Performs data validation, duplicate checking, and assistant-masked tensor generation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset

from app.ml.fine_tuning.config import FineTuningConfig
from app.ml.fine_tuning.exceptions import DatasetValidationError
from app.ml.fine_tuning.validator import ConversationValidator
from app.ml.fine_tuning.loss_mask import AssistantLossMaskBuilder
from app.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class ConversationalFineTuningDataset(Dataset):
    """
    Dataset loader for conversational fine-tuning JSONL files.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Tokenizer,
        config: FineTuningConfig,
        validator: Optional[ConversationValidator] = None
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.config = config
        self.validator = validator or ConversationValidator()
        self.mask_builder = AssistantLossMaskBuilder(
            tokenizer=tokenizer,
            config=None,
            ignore_index=config.ignore_index,
            max_sequence_length=config.max_sequence_length
        )

        self.examples: List[List[Dict[str, str]]] = []
        self.statistics: Dict[str, Any] = {}

        self._load_and_validate()

    def _load_and_validate(self) -> None:
        """Loads JSONL dataset, validates records, deduplicates, and compiles statistics."""
        if not self.jsonl_path.exists():
            raise DatasetValidationError(f"Dataset file not found at {self.jsonl_path}")

        total_examples = 0
        valid_examples = 0
        invalid_examples = 0
        quarantined_examples = 0
        duplicate_examples = 0

        total_turns = 0
        user_len_sum = 0
        asst_len_sum = 0
        max_conv_len = 0

        seen_fingerprints = set()

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue

                total_examples += 1
                try:
                    raw_record = json.loads(line_str)
                except json.JSONDecodeError:
                    invalid_examples += 1
                    quarantined_examples += 1
                    continue

                is_valid, reason, messages = self.validator.validate_conversation(raw_record)
                if not is_valid:
                    invalid_examples += 1
                    quarantined_examples += 1
                    logger.debug(f"Line {line_idx} quarantined: {reason}")
                    continue

                # Deduplication fingerprint based on normalized turn contents
                fp = "||".join(f"{m['role']}:{m['content']}" for m in messages)
                if fp in seen_fingerprints:
                    duplicate_examples += 1
                    continue

                seen_fingerprints.add(fp)
                valid_examples += 1
                self.examples.append(messages)

                # Statistics collection
                turns = len(messages)
                total_turns += turns
                if turns > max_conv_len:
                    max_conv_len = turns

                for m in messages:
                    c_len = len(m["content"])
                    if m["role"] == "user":
                        user_len_sum += c_len
                    elif m["role"] == "assistant":
                        asst_len_sum += c_len

        self.statistics = {
            "dataset_path": str(self.jsonl_path),
            "total_examples": total_examples,
            "valid_examples": valid_examples,
            "invalid_examples": invalid_examples,
            "quarantined_examples": quarantined_examples,
            "duplicate_examples": duplicate_examples,
            "average_turns": (total_turns / valid_examples) if valid_examples else 0.0,
            "average_user_length": (user_len_sum / valid_examples) if valid_examples else 0.0,
            "average_assistant_length": (asst_len_sum / valid_examples) if valid_examples else 0.0,
            "maximum_conversation_length": max_conv_len,
        }

        logger.info(
            f"Loaded dataset {self.jsonl_path.name}: "
            f"{valid_examples}/{total_examples} valid records loaded ({duplicate_examples} duplicates skipped)."
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        messages = self.examples[idx]
        return self.mask_builder.build_example(messages)
