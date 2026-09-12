"""
Precious Edu LLM — Tokenized Data Pipeline & DataLoader

Implements PyTorch Dataset for pretokenized JSONL files with causal input/target shifting,
sequence chunking, padding, vocabulary bounds validation, and DataLoader creation.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader

from app.ml.training.exceptions import DataLoadingError


class TokenizedDataset(Dataset):
    """
    Dataset class for reading pretokenized language sequences.

    Input sequence shifting for Causal LM pretraining:
    tokens = [t0, t1, t2, ..., t_N]
    input_ids  = [t0, t1, t2, ..., t_{N-1}]
    target_ids = [t1, t2, t3, ..., t_N]
    """

    def __init__(
        self,
        jsonl_path: Path,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
        vocab_size: int = 175,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        if not self.jsonl_path.exists():
            raise DataLoadingError(f"Tokenized dataset file not found at: {self.jsonl_path}")

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._load_and_process_file()

    def _load_and_process_file(self) -> None:
        """Read JSONL file line-by-line, chunk/pad sequences, and construct shifted pairs."""
        target_token_count = self.max_seq_len + 1  # Need max_seq_len + 1 tokens to shift input/target

        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    raw_ids = data.get("input_ids", [])
                    
                    if not raw_ids:
                        continue

                    # Validate token IDs range
                    for tok_id in raw_ids:
                        if not (0 <= tok_id < self.vocab_size):
                            raise DataLoadingError(
                                f"Token ID {tok_id} at line {line_idx} is out of vocabulary bounds (0-{self.vocab_size-1})"
                            )

                    # Chunk long sequences into contiguous segments of target_token_count
                    if len(raw_ids) >= target_token_count:
                        for idx in range(0, len(raw_ids) - target_token_count + 1, target_token_count):
                            segment = raw_ids[idx : idx + target_token_count]
                            self._add_shifted_sample(segment)
                    else:
                        # Pad short sequences with pad_token_id
                        padded = raw_ids + [self.pad_token_id] * (target_token_count - len(raw_ids))
                        self._add_shifted_sample(padded)

        except Exception as e:
            if isinstance(e, DataLoadingError):
                raise
            raise DataLoadingError(f"Failed to parse dataset {self.jsonl_path}: {e}")

        if len(self.samples) == 0:
            raise DataLoadingError(f"No valid training samples produced from dataset {self.jsonl_path}")

    def _add_shifted_sample(self, token_segment: List[int]) -> None:
        """Create shifted (input_ids, target_ids) pair."""
        inputs = torch.tensor(token_segment[:-1], dtype=torch.long)
        targets = torch.tensor(token_segment[1:], dtype=torch.long)
        self.samples.append((inputs, targets))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def create_dataloader(
    dataset: TokenizedDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Construct PyTorch DataLoader for TokenizedDataset.

    Args:
        dataset: TokenizedDataset instance.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle sequences.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, copies tensors to CUDA pinned memory before returning.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
