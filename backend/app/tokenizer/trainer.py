"""
Precious Edu LLM — BPE Tokenizer Trainer

Trains custom BPE tokenizer from Phase 4 training data (train.jsonl only).
Strictly enforces data-leakage rules to prevent validation/test set contamination.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional
from app.tokenizer.config import TokenizerConfig, get_tokenizer_config
from app.tokenizer.exceptions import TokenizerTrainingError
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.bpe import BPEAlgorithm
from app.tokenizer.formatter import ConversationFormatter
from app.dataset.models import DatasetRecord, RecordType, MessageRecord

logger = logging.getLogger(__name__)


class BPETrainer:
    """
    Trainer for BPE Tokenizer.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or get_tokenizer_config()
        self.formatter = ConversationFormatter(self.config)
        self.bpe = BPEAlgorithm(special_tokens=set(self.config.special_tokens_list))

    def train_from_file(self, train_file: Path) -> Tuple[Vocabulary, List[Tuple[str, str]]]:
        """
        Train BPE tokenizer on training dataset file.

        Enforces strict data leakage rule: train_file MUST be train.jsonl.
        """
        self._verify_no_data_leakage(train_file)

        logger.info(f"Loading training data from {train_file} for BPE tokenizer training...")
        formatted_texts = self._load_and_format_training_corpus(train_file)

        if not formatted_texts:
            raise TokenizerTrainingError(f"Training corpus in {train_file} is empty or invalid.")

        logger.info(f"Loaded {len(formatted_texts)} formatted training sequences.")

        return self.train_from_sequences(formatted_texts)

    def train_from_sequences(self, sequences_text: List[str]) -> Tuple[Vocabulary, List[Tuple[str, str]]]:
        """
        Execute BPE training loop on text sequences.
        """
        # 1. Initialize Vocabulary with special tokens
        vocab = Vocabulary()
        for st_token, st_id in self.config.special_tokens_map.items():
            vocab.add_token(st_token, forced_id=st_id)

        # 2. Tokenize sequences into initial symbols
        sequences = [self.bpe.tokenize_to_symbols(txt) for txt in sequences_text]

        # 3. Add all unique initial base characters/symbols to vocabulary
        base_symbols: Set[str] = set()
        for seq in sequences:
            for sym in seq:
                if sym not in vocab:
                    base_symbols.add(sym)

        # Sort base symbols deterministically before adding
        for sym in sorted(base_symbols):
            vocab.add_token(sym)

        logger.info(f"Initial vocabulary size (special tokens + base symbols): {len(vocab)}")

        # 4. BPE Merge Loop
        merges: List[Tuple[str, str]] = []
        target_vocab_size = self.config.VOCAB_SIZE
        max_merges = self.config.MAX_MERGES
        min_freq = self.config.MIN_PAIR_FREQUENCY

        while len(vocab) < target_vocab_size and len(merges) < max_merges:
            # Count adjacent pair frequencies
            pair_counts = self.bpe.get_pair_frequencies(sequences)
            if not pair_counts:
                logger.info("No more valid merge pairs found. Stopping BPE training.")
                break

            # Select best pair deterministically
            best_pair = self.bpe.get_best_pair(pair_counts)
            if not best_pair:
                break

            best_freq = pair_counts[best_pair]
            if best_freq < min_freq:
                logger.info(f"Highest pair frequency ({best_freq}) fell below MIN_PAIR_FREQUENCY ({min_freq}). Stopping BPE training.")
                break

            new_token = best_pair[0] + best_pair[1]

            # Add new merged token to vocabulary
            vocab.add_token(new_token)
            merges.append(best_pair)

            # Apply merge to active sequences
            sequences = self.bpe.merge_pair_in_sequences(sequences, best_pair, new_token)

            if len(vocab) % 500 == 0:
                logger.info(f"BPE training progress: Vocab size = {len(vocab)} / {target_vocab_size}")

        vocab.validate_integrity()
        logger.info(f"BPE training complete. Final Vocab Size: {len(vocab)}, Total Merges: {len(merges)}")

        return vocab, merges

    def _verify_no_data_leakage(self, file_path: Path) -> None:
        """
        Verify input path is strictly train.jsonl.
        Rejects validation.jsonl, test.jsonl, or evaluation/ paths to prevent data leakage.
        """
        path_str = str(file_path).lower().replace("\\", "/")
        if "validation" in path_str or "test.jsonl" in path_str or "evaluation" in path_str:
            raise TokenizerTrainingError(
                f"Data leakage rule violation: Tokenizer training MUST use train.jsonl strictly. Rejecting path: {file_path}"
            )

    def _load_and_format_training_corpus(self, train_file: Path) -> List[str]:
        """Load and format JSONL records into strings."""
        texts: List[str] = []
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    if "messages" in data and isinstance(data["messages"], list):
                        msgs = [MessageRecord(role=m.get("role", ""), content=m.get("content", "")) for m in data["messages"]]
                        rec = DatasetRecord(record_id="dummy", source_id="train", type=RecordType.CONVERSATION, messages=msgs)
                        formatted = self.formatter.format_record(rec)
                        if formatted:
                            texts.append(formatted)
                    elif "text" in data and isinstance(data["text"], str):
                        rec = DatasetRecord(record_id="dummy", source_id="train", type=RecordType.PLAIN_TEXT, text=data["text"])
                        formatted = self.formatter.format_record(rec)
                        if formatted:
                            texts.append(formatted)
                except Exception as err:
                    logger.warning(f"Error parsing training line: {err}")

        return texts
