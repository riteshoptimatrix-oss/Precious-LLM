"""
Precious Edu LLM — Fine-Tune Model CLI Script

Main CLI script for executing Phase 8 conversational fine-tuning.
Supports resume from checkpoint, dry-run testing, configuration overrides,
and automatic evaluation.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add backend directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.fine_tuning.config import FineTuningConfig
from app.ml.fine_tuning.dataset import ConversationalFineTuningDataset
from app.ml.fine_tuning.trainer import FineTuningEngine
from app.ml.training.reproducibility import set_seed
from app.tokenizer.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Precious AI Conversational Fine-Tuning CLI.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to fine-tuning checkpoint to resume from.")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None, help="Path to pretrained base model checkpoint.")
    parser.add_argument("--train_dataset", type=str, default="data/training/conversational/train.jsonl", help="Path to train split JSONL.")
    parser.add_argument("--val_dataset", type=str, default="data/training/conversational/validation.jsonl", help="Path to validation split JSONL.")
    parser.add_argument("--max_sequence_length", type=int, default=64, help="Override maximum sequence length.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum training steps.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Perform single-step dry-run verification.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # 1. Load or build config
    if args.config:
        config = FineTuningConfig.from_yaml(args.config)
    else:
        config = FineTuningConfig()

    if args.pretrained_checkpoint:
        config.pretrained_checkpoint_path = args.pretrained_checkpoint
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_sequence_length:
        config.max_sequence_length = args.max_sequence_length
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.epochs:
        config.epochs = args.epochs

    if args.dry_run:
        config.max_steps = 5
        config.epochs = 1
        config.evaluation_interval = 2
        config.checkpoint_interval = 5
        config.logging_interval = 1

    config.seed = args.seed
    config.validate()

    # 2. Set Reproducibility Seed
    set_seed(config.seed)

    # 3. Load Tokenizer
    tokenizer_path = Path(config.tokenizer_dir)
    if not tokenizer_path.exists():
        fallback_path = PROJECT_ROOT.parent / config.tokenizer_dir
        if fallback_path.exists():
            tokenizer_path = fallback_path
        else:
            logger.error(f"Tokenizer directory not found: {tokenizer_path}")
            sys.exit(1)
    tokenizer = Tokenizer.load(tokenizer_path)

    # 4. Load Datasets
    train_path = Path(args.train_dataset)
    if not train_path.exists() and (PROJECT_ROOT.parent / args.train_dataset).exists():
        train_path = PROJECT_ROOT.parent / args.train_dataset

    logger.info(f"Loading train dataset from: {train_path}")
    train_ds = ConversationalFineTuningDataset(
        jsonl_path=str(train_path),
        tokenizer=tokenizer,
        config=config,
    )

    val_ds = None
    val_path = Path(args.val_dataset)
    if not val_path.exists() and (PROJECT_ROOT.parent / args.val_dataset).exists():
        val_path = PROJECT_ROOT.parent / args.val_dataset

    if val_path.exists():
        logger.info(f"Loading validation dataset from: {val_path}")
        val_ds = ConversationalFineTuningDataset(
            jsonl_path=str(val_path),
            tokenizer=tokenizer,
            config=config,
        )

    # 5. Build Trainer Engine
    engine = FineTuningEngine(
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    if args.resume:
        engine.resume_from_checkpoint(args.resume)

    # 6. Execute Fine-Tuning
    results = engine.train()
    logger.info(f"Fine-Tuning completed successfully: {results}")


if __name__ == "__main__":
    main()
