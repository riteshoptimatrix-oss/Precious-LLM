"""
Precious Edu LLM — Training Execution Script

CLI script to launch or resume LLM pretraining runs.

Usage:
    python -m scripts.train_model [--dry-run] [--resume <checkpoint_path>] [--config <config_yaml>]
"""

import sys
import argparse
from pathlib import Path

from app.tokenizer import Tokenizer
from app.ml.model import ModelConfig, PreciousTransformer
from app.ml.training import (
    TrainingConfig,
    TokenizedDataset,
    create_dataloader,
    PretrainingEngine,
)


def main():
    parser = argparse.ArgumentParser(description="Precious AI LLM Pretraining CLI")
    parser.add_argument("--dry-run", action="store_true", help="Execute pre-flight check only")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume training from")
    parser.add_argument("--config", type=str, default=None, help="Path to optional training config YAML file")
    args = parser.parse_args()

    # 1. Load Training Config
    if args.config:
        training_config = TrainingConfig.from_yaml(args.config)
    else:
        training_config = TrainingConfig(
            max_sequence_length=64,
            total_steps=20,
            epochs=10,
            evaluation_interval=10,
            checkpoint_interval=10,
            logging_interval=5,
        )

    training_config.validate()

    # 2. Locate and Load Frozen Tokenizer
    tokenizer_dir = Path("artifacts") / "tokenizer" / "v1"
    if not tokenizer_dir.exists():
        tokenizer_dir = Path("..") / "artifacts" / "tokenizer" / "v1"

    if not tokenizer_dir.exists():
        print(f"Error: Tokenizer artifacts not found at {tokenizer_dir}")
        print("Please run 'python -m scripts.train_tokenizer' first.")
        sys.exit(1)

    tokenizer = Tokenizer.load(tokenizer_dir)
    st_map = tokenizer.config.special_tokens_map

    # 3. Construct ModelConfig matching tokenizer
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=st_map["<pad>"],
        bos_token_id=st_map["<bos>"],
        eos_token_id=st_map["<eos>"],
        unk_token_id=st_map["<unk>"],
        max_seq_length=training_config.max_sequence_length,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
        attention_dropout=0.1,
        weight_tying=True,
    )
    model_config.validate()

    # 4. Instantiate Model
    model = PreciousTransformer(model_config)

    # 5. Load Dataset Files
    tokenized_dir = Path(training_config.dataset_dir)
    if not tokenized_dir.exists():
        tokenized_dir = Path("..") / "data" / "processed" / "tokenized"
    if not tokenized_dir.exists():
        tokenized_dir = Path("data") / "processed" / "tokenized"

    train_jsonl = tokenized_dir / "train_tokenized.jsonl"
    val_jsonl = tokenized_dir / "validation_tokenized.jsonl"

    if not train_jsonl.exists() or not val_jsonl.exists():
        print(f"Error: Tokenized datasets not found under {tokenized_dir}")
        print("Please run 'python -m scripts.tokenize_dataset' first.")
        sys.exit(1)

    train_dataset = TokenizedDataset(
        jsonl_path=train_jsonl,
        max_seq_len=training_config.max_sequence_length,
        pad_token_id=st_map["<pad>"],
        vocab_size=tokenizer.vocab_size,
    )

    val_dataset = TokenizedDataset(
        jsonl_path=val_jsonl,
        max_seq_len=training_config.max_sequence_length,
        pad_token_id=st_map["<pad>"],
        vocab_size=tokenizer.vocab_size,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    # 6. Create Run Directory
    run_id = f"run-{Path(__file__).stem}"
    run_dir = Path(training_config.artifacts_dir) / run_id

    engine = PretrainingEngine(
        config=training_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
    )

    # 7. Run Dry-Run or Full Training
    if args.dry_run:
        engine.dry_run()
    else:
        engine.train(resume_checkpoint_path=args.resume)


if __name__ == "__main__":
    main()
