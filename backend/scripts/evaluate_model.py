"""
Precious Edu LLM — Model Evaluation Script

CLI script to evaluate a saved model checkpoint on validation or test dataset splits.

Usage:
    python -m scripts.evaluate_model --checkpoint artifacts/training/run-train_model/checkpoints/latest.pt --split validation
"""

import sys
import argparse
from pathlib import Path
import torch

from app.tokenizer import Tokenizer
from app.ml.model import PreciousTransformer, ModelConfig
from app.ml.training import TokenizedDataset, create_dataloader, Evaluator, load_training_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Precious AI Model Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pt)")
    parser.add_argument("--split", type=str, choices=["validation", "test"], default="validation", help="Dataset split")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    # 1. Load Tokenizer
    tokenizer_dir = Path("artifacts") / "tokenizer" / "v1"
    if not tokenizer_dir.exists():
        tokenizer_dir = Path("..") / "artifacts" / "tokenizer" / "v1"

    tokenizer = Tokenizer.load(tokenizer_dir)
    st_map = tokenizer.config.special_tokens_map

    # 2. Build Model matching saved checkpoint config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig.from_dict(ckpt_data["model_config"])
    model = PreciousTransformer(model_cfg)

    step, epoch, loss, metrics, model_config_dict, _ = load_training_checkpoint(
        str(checkpoint_path), model=model, device=device
    )

    # 3. Load Selected Dataset Split
    data_dir = Path("..") / "data" / "processed" / "tokenized"
    if not data_dir.exists():
        data_dir = Path("data") / "processed" / "tokenized"
    jsonl_filename = f"{args.split}_tokenized.jsonl"
    split_path = data_dir / jsonl_filename

    if not split_path.exists():
        print(f"Error: Dataset split file not found: {split_path}")
        sys.exit(1)

    dataset = TokenizedDataset(
        jsonl_path=split_path,
        max_seq_len=model.config.max_seq_length,
        pad_token_id=st_map["<pad>"],
        vocab_size=tokenizer.vocab_size,
    )
    loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    # 4. Evaluate
    evaluator = Evaluator(model, loader, device)
    eval_loss, eval_ppl = evaluator.evaluate()

    print("\n========================================")
    print(f"PRECIOUS AI MODEL EVALUATION ({args.split.upper()} SPLIT)")
    print("========================================")
    print(f"Checkpoint Path    : {checkpoint_path}")
    print(f"Checkpoint Step    : {step}")
    print(f"Checkpoint Epoch   : {epoch}")
    print(f"Dataset Split      : {args.split}")
    print(f"Evaluated Samples  : {len(dataset)}")
    print(f"Evaluation Loss    : {eval_loss:.4f}")
    print(f"Evaluation PPL     : {eval_ppl:.2f}")
    print("========================================\n")


if __name__ == "__main__":
    main()
