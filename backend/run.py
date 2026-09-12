"""
Precious Edu LLM — Unified Run & Training CLI

Unified entry point for running the application, training the custom LLM,
preparing datasets, and managing knowledge ingestion.

Usage:
    # 1. Start Backend Server (Default)
    python run.py
    python run.py server [--host 0.0.0.0] [--port 8000] [--reload]

    # 2. Train Custom LLM (Full Pipeline or Specific Steps)
    python run.py train-all            # Run complete end-to-end training pipeline
    python run.py prepare-data         # Step 1: Clean & prepare raw dataset (train/val/test splits)
    python run.py train-tokenizer      # Step 2: Train custom BPE tokenizer
    python run.py tokenize-data        # Step 3: Tokenize dataset for model pretraining
    python run.py train-base           # Step 4: Train base Transformer model from scratch
    python run.py train-domain         # Step 5: Domain adaptation training on Precious Edu data
    python run.py finetune             # Step 6: Conversational fine-tuning

    # 3. Model Evaluation
    python run.py evaluate             # Evaluate fine-tuned conversational model
    python run.py evaluate --model base # Evaluate base model
    python run.py evaluate --model domain # Evaluate domain model

    # 4. Knowledge Management
    python run.py import-qa            # Import structured Q&A into MongoDB (idempotent)
    python run.py import-qa --stats-only # Inspect dataset statistics without importing
    python run.py crawl-website        # Crawl website into website_chunks
    python run.py refresh-website      # Refresh updated website pages
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional


BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def run_module(module_name: str, extra_args: Optional[List[str]] = None) -> int:
    """Executes a python module with arguments from the backend directory."""
    cmd = [sys.executable, "-m", module_name] + (extra_args or [])
    print(f"\n[CLI] Executing: {' '.join(cmd)}")
    print("-" * 65)
    try:
        proc = subprocess.run(cmd, cwd=str(BACKEND_DIR))
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[CLI] Process interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n[CLI] Error executing {module_name}: {e}")
        return 1


def start_server(host: Optional[str] = None, port: Optional[int] = None, reload: Optional[bool] = None) -> None:
    """Starts the FastAPI application via Uvicorn."""
    import uvicorn
    from app.config import get_settings

    settings = get_settings()
    server_host = host or settings.APP_HOST
    server_port = port or settings.APP_PORT
    server_reload = reload if reload is not None else settings.APP_DEBUG

    print("=" * 65)
    print("  PRECIOUS EDU LLM — BACKEND SERVER")
    print(f"  Host: {server_host} | Port: {server_port} | Reload: {server_reload}")
    print("=" * 65)

    uvicorn.run(
        "app.main:app",
        host=server_host,
        port=server_port,
        reload=server_reload,
        log_level=settings.LOG_LEVEL.lower(),
    )


def run_train_all(extra_args: Optional[List[str]] = None) -> int:
    """Runs the full end-to-end LLM training pipeline."""
    print("=" * 65)
    print("  PRECIOUS EDU LLM — COMPLETE TRAINING PIPELINE")
    print("=" * 65)

    pipeline_stages = [
        ("Step 1/6: Prepare Training Dataset", "scripts.prepare_dataset", []),
        ("Step 2/6: Train BPE Tokenizer", "scripts.train_tokenizer", []),
        ("Step 3/6: Tokenize Dataset", "scripts.tokenize_dataset", []),
        ("Step 4/6: Train Base Transformer Model", "scripts.train_model", extra_args or []),
        ("Step 5/6: Domain Adaptation Training", "scripts.train_domain", []),
        ("Step 6/6: Conversational Fine-Tuning", "scripts.finetune_model", []),
    ]

    for title, module, args in pipeline_stages:
        print(f"\n>>> Running {title}...")
        rc = run_module(module, args)
        if rc != 0:
            print(f"\n[ERROR] Pipeline failed at {title} with exit code {rc}")
            return rc

    print("\n" + "=" * 65)
    print("  TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precious Edu LLM — Unified Run & Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                               # Start server
  python run.py train-all                     # Run full training pipeline
  python run.py train-tokenizer               # Train tokenizer only
  python run.py train-base --epochs 10        # Train base model with 10 epochs
  python run.py train-domain --lr 0.0001      # Domain adaptation training
  python run.py finetune --epochs 5           # Fine-tune conversational model
  python run.py import-qa --stats-only        # View structured Q&A statistics
        """
    )

    # Subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 1. Server command
    server_p = subparsers.add_parser("server", help="Start the FastAPI backend server (default)")
    server_p.add_argument("--host", type=str, default=None, help="Server host (default: from config)")
    server_p.add_argument("--port", type=int, default=None, help="Server port (default: from config)")
    server_p.add_argument("--reload", action="store_true", default=None, help="Enable auto-reload")

    # 2. Training commands
    subparsers.add_parser("train-all", help="Execute complete end-to-end training pipeline")
    subparsers.add_parser("prepare-data", help="Prepare & split raw dataset (train/val/test)")
    subparsers.add_parser("train-tokenizer", help="Train custom BPE tokenizer from training dataset")
    subparsers.add_parser("tokenize-data", help="Tokenize training data for pretraining")

    # Base model training
    base_p = subparsers.add_parser("train-base", aliases=["train-model"], help="Train base Transformer model")
    base_p.add_argument("--dry-run", action="store_true", help="Dry run verification only")
    base_p.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume")
    base_p.add_argument("--config", type=str, default=None, help="Path to training config YAML")

    # Domain adaptation training
    domain_p = subparsers.add_parser("train-domain", help="Train domain-adapted model on Precious Edu knowledge")
    domain_p.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    domain_p.add_argument("--overfit-test", action="store_true", help="Run tiny overfit test")
    domain_p.add_argument("--epochs", type=int, default=None, help="Override epochs")
    domain_p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    domain_p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    domain_p.add_argument("--base-checkpoint", type=str, default=None, help="Path to base model checkpoint")

    # Fine-tuning
    ft_p = subparsers.add_parser("finetune", aliases=["finetune-model"], help="Run conversational fine-tuning")
    ft_p.add_argument("--config", type=str, default=None, help="Path to config YAML")
    ft_p.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    ft_p.add_argument("--pretrained_checkpoint", type=str, default=None, help="Pretrained base model checkpoint")
    ft_p.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    ft_p.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    ft_p.add_argument("--dry-run", action="store_true", help="Dry-run verification")

    # 3. Evaluation
    eval_p = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_p.add_argument(
        "--model",
        type=str,
        default="finetuned",
        choices=["base", "finetuned", "domain", "safety", "website"],
        help="Which model/evaluation to run"
    )

    # 4. Knowledge Management
    qa_p = subparsers.add_parser("import-qa", help="Import structured Q&A dataset into MongoDB")
    qa_p.add_argument("--stats-only", action="store_true", help="Inspect dataset statistics without importing")
    qa_p.add_argument("--data-dir", type=str, default=None, help="Custom data directory")

    crawl_p = subparsers.add_parser("crawl-website", help="Crawl website knowledge into MongoDB")
    crawl_p.add_argument("--dry-run", action="store_true", help="Discover URLs without storing")

    subparsers.add_parser("refresh-website", help="Refresh changed website pages")

    # Convenience flags for users who prefer --train, etc.
    parser.add_argument("--train", action="store_true", help="Alias for train-all")
    parser.add_argument("--train-tokenizer", action="store_true", help="Alias for train-tokenizer")
    parser.add_argument("--train-base", action="store_true", help="Alias for train-base")
    parser.add_argument("--train-domain", action="store_true", help="Alias for train-domain")
    parser.add_argument("--finetune", action="store_true", help="Alias for finetune")
    parser.add_argument("--import-qa", action="store_true", help="Alias for import-qa")
    parser.add_argument("--server", action="store_true", help="Explicitly start server")

    return parser


def main():
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    # Handle convenience flags
    if args.train:
        sys.exit(run_train_all(unknown))
    elif args.train_tokenizer:
        sys.exit(run_module("scripts.train_tokenizer", unknown))
    elif args.train_base:
        sys.exit(run_module("scripts.train_model", unknown))
    elif args.train_domain:
        sys.exit(run_module("scripts.train_domain", unknown))
    elif args.finetune:
        sys.exit(run_module("scripts.finetune_model", unknown))
    elif args.import_qa:
        sys.exit(run_module("scripts.import_structured_qa", unknown))
    elif args.server:
        start_server()
        return

    # Handle subcommands
    cmd = args.command

    if cmd is None or cmd == "server":
        # Default action: start server
        host = getattr(args, "host", None)
        port = getattr(args, "port", None)
        reload = getattr(args, "reload", None)
        start_server(host=host, port=port, reload=reload)

    elif cmd == "train-all":
        sys.exit(run_train_all(unknown))

    elif cmd == "prepare-data":
        sys.exit(run_module("scripts.prepare_dataset", unknown))

    elif cmd == "train-tokenizer":
        sys.exit(run_module("scripts.train_tokenizer", unknown))

    elif cmd == "tokenize-data":
        sys.exit(run_module("scripts.tokenize_dataset", unknown))

    elif cmd in ("train-base", "train-model"):
        forward_args = []
        if getattr(args, "dry_run", False):
            forward_args.append("--dry-run")
        if getattr(args, "resume", None):
            forward_args.extend(["--resume", args.resume])
        if getattr(args, "config", None):
            forward_args.extend(["--config", args.config])
        forward_args.extend(unknown)
        sys.exit(run_module("scripts.train_model", forward_args))

    elif cmd == "train-domain":
        forward_args = []
        if getattr(args, "resume", None):
            forward_args.extend(["--resume", args.resume])
        if getattr(args, "overfit_test", False):
            forward_args.append("--overfit-test")
        if getattr(args, "epochs", None) is not None:
            forward_args.extend(["--epochs", str(args.epochs)])
        if getattr(args, "lr", None) is not None:
            forward_args.extend(["--lr", str(args.lr)])
        if getattr(args, "batch_size", None) is not None:
            forward_args.extend(["--batch-size", str(args.batch_size)])
        if getattr(args, "base_checkpoint", None):
            forward_args.extend(["--base-checkpoint", args.base_checkpoint])
        forward_args.extend(unknown)
        sys.exit(run_module("scripts.train_domain", forward_args))

    elif cmd in ("finetune", "finetune-model"):
        forward_args = []
        if getattr(args, "config", None):
            forward_args.extend(["--config", args.config])
        if getattr(args, "resume", None):
            forward_args.extend(["--resume", args.resume])
        if getattr(args, "pretrained_checkpoint", None):
            forward_args.extend(["--pretrained_checkpoint", args.pretrained_checkpoint])
        if getattr(args, "epochs", None) is not None:
            forward_args.extend(["--epochs", str(args.epochs)])
        if getattr(args, "learning_rate", None) is not None:
            forward_args.extend(["--learning_rate", str(args.learning_rate)])
        if getattr(args, "dry_run", False):
            forward_args.append("--dry-run")
        forward_args.extend(unknown)
        sys.exit(run_module("scripts.finetune_model", forward_args))

    elif cmd == "evaluate":
        model_type = getattr(args, "model", "finetuned")
        eval_script_map = {
            "base": "scripts.evaluate_model",
            "finetuned": "scripts.evaluate_finetuned_model",
            "domain": "scripts.evaluate_domain",
            "safety": "scripts.evaluate_safety",
            "website": "scripts.evaluate_website_knowledge",
        }
        script = eval_script_map.get(model_type, "scripts.evaluate_finetuned_model")
        sys.exit(run_module(script, unknown))

    elif cmd == "import-qa":
        forward_args = []
        if getattr(args, "stats_only", False):
            forward_args.append("--stats-only")
        if getattr(args, "data_dir", None):
            forward_args.extend(["--data-dir", args.data_dir])
        forward_args.extend(unknown)
        sys.exit(run_module("scripts.import_structured_qa", forward_args))

    elif cmd == "crawl-website":
        forward_args = []
        if getattr(args, "dry_run", False):
            forward_args.append("--dry-run")
        forward_args.extend(unknown)
        sys.exit(run_module("scripts.crawl_website", forward_args))

    elif cmd == "refresh-website":
        sys.exit(run_module("scripts.refresh_website", unknown))


if __name__ == "__main__":
    main()
