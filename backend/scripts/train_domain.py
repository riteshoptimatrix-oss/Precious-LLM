"""
Script to train the domain-adapted Precious AI LLM.

Usage:
    python -m scripts.train_domain [--resume CHECKPOINT] [--overfit-test] [--epochs EPOCHS] [--lr LR]
"""

import argparse
import sys
import logging
from pathlib import Path

# Add backend root to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.domain_training.config import DomainTrainingConfig
from app.domain_training.trainer import DomainTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Domain-Adapted Precious AI LLM")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--overfit-test", action="store_true", help="Run tiny overfit test to verify pipeline correctness")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epochs count")
    parser.add_argument("--lr", type=float, default=None, help="Override default learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--base-checkpoint", type=str, default=None, help="Path to base Phase 8 model checkpoint")

    args = parser.parse_args()

    config = DomainTrainingConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.base_checkpoint is not None:
        config.base_model_checkpoint = args.base_checkpoint

    trainer = DomainTrainer(config=config)

    if args.overfit_test:
        logger.info("Starting tiny overfit test...")
        results = trainer.run_overfit_test()
        logger.info(f"Overfit test finished. Passed: {results['passed']}, Loss reduction: {results['loss_reduction']:.4f}")
        if not results['passed']:
            sys.exit(1)
    else:
        logger.info("Starting domain fine-tuning...")
        train_result = trainer.train(resume_checkpoint=args.resume)
        final_model_path = train_result.get("final_model_path", train_result) if isinstance(train_result, dict) else train_result
        logger.info(f"Domain training completed successfully! Final model: {final_model_path}")

if __name__ == "__main__":
    main()
