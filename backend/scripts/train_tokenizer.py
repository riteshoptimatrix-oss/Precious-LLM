"""
Precious Edu LLM — Train Tokenizer Script

Trains custom BPE tokenizer from Phase 4 training data (train.jsonl only).
Saves artifacts to artifacts/tokenizer/v1/ and prints summary.
"""

import sys
import logging
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dataset.config import get_dataset_config
from app.tokenizer.config import get_tokenizer_config
from app.tokenizer.trainer import BPETrainer
from app.tokenizer.tokenizer import Tokenizer
from app.tokenizer.statistics import TokenizerStatisticsCalculator


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    print("==================================================")
    print("Precious AI — Training Custom BPE Tokenizer")
    print("==================================================")

    data_config = get_dataset_config()
    tok_config = get_tokenizer_config()

    train_file = data_config.TRAINING_DIR / "train.jsonl"
    if not train_file.exists():
        print(f"Error: Training file not found at {train_file}")
        print("Please run 'python -m scripts.prepare_dataset' first.")
        sys.exit(1)

    trainer = BPETrainer(config=tok_config)

    # 1. Train BPE Tokenizer (strictly on train.jsonl)
    vocab, merges = trainer.train_from_file(train_file)

    tokenizer = Tokenizer(vocab=vocab, merges=merges, config=tok_config)

    # 2. Calculate Statistics
    raw_texts = trainer._load_and_format_training_corpus(train_file)
    encoded_seqs = tokenizer.encode_batch(raw_texts)

    stats_calc = TokenizerStatisticsCalculator(tok_config)
    stats = stats_calc.calculate_statistics(
        raw_texts=raw_texts,
        token_id_sequences=encoded_seqs,
        vocab_size=len(vocab),
        merges_count=len(merges)
    )

    # 3. Save Artifacts
    output_dir = tok_config.ensure_artifacts_dir()
    tokenizer.save(
        output_dir=output_dir,
        dataset_version=data_config.DATASET_VERSION,
        statistics=stats
    )

    print("\n--------------------------------------------------")
    print("Tokenizer Training Summary:")
    print("--------------------------------------------------")
    print(f"Algorithm:           {tok_config.TOKENIZER_TYPE.upper()}")
    print(f"Tokenizer Version:   {tok_config.TOKENIZER_VERSION}")
    print(f"Dataset Used:        train.jsonl (v{data_config.DATASET_VERSION})")
    print(f"Training Sequences:  {stats.get('training_records_count')}")
    print(f"Vocabulary Size:     {stats.get('vocab_size')}")
    print(f"Merges Learned:      {stats.get('merges_count')}")
    print(f"Total Tokens:        {stats.get('total_tokens')}")
    print(f"Compression Ratio:   {stats.get('compression_ratio_chars_per_token')} chars/token")
    print(f"Unknown Token Rate:  {stats.get('unknown_token_rate_pct')}%")
    print("--------------------------------------------------")
    print(f"Artifacts saved to: {output_dir}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
