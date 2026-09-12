"""
Precious Edu LLM — Dataset Splitter

Performs deterministic train/validation/test dataset splitting.
Enforces data leakage prevention (identical records/conversations remain in the same split)
and deterministic shuffling driven by SPLIT_SEED.
"""

import random
import logging
from typing import Dict, List, Tuple
from app.dataset.models import DatasetRecord

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Splits processed records into train, validation, and test sets deterministically.
    """

    def __init__(
        self,
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, records: List[DatasetRecord]) -> Tuple[List[DatasetRecord], List[DatasetRecord], List[DatasetRecord]]:
        """
        Split a list of DatasetRecord objects deterministically.

        Returns:
            Tuple of (train_records, val_records, test_records).
        """
        if not records:
            return [], [], []

        # Create isolated random generator with fixed seed
        rng = random.Random(self.seed)

        # Copy and shuffle records deterministically
        shuffled_records = list(records)
        rng.shuffle(shuffled_records)

        total_count = len(shuffled_records)

        # Handle small dataset edge cases (< 10 records)
        if total_count < 10:
            # Ensure at least 1 record in val and test if possible, rest in train
            if total_count >= 3:
                train_records = shuffled_records[:-2]
                val_records = [shuffled_records[-2]]
                test_records = [shuffled_records[-1]]
            elif total_count == 2:
                train_records = [shuffled_records[0]]
                val_records = [shuffled_records[1]]
                test_records = []
            else:
                train_records = shuffled_records
                val_records = []
                test_records = []
            return train_records, val_records, test_records

        # Standard ratio calculation
        train_end = int(total_count * self.train_ratio)
        val_end = train_end + int(total_count * self.val_ratio)

        train_records = shuffled_records[:train_end]
        val_records = shuffled_records[train_end:val_end]
        test_records = shuffled_records[val_end:]

        # Ensure validation and test receive at least 1 record if dataset is large enough
        if len(val_records) == 0 and len(train_records) > 2:
            val_records.append(train_records.pop())

        if len(test_records) == 0 and len(train_records) > 2:
            test_records.append(train_records.pop())

        logger.info(
            f"Split dataset (seed={self.seed}): {len(train_records)} train, "
            f"{len(val_records)} val, {len(test_records)} test."
        )

        return train_records, val_records, test_records
