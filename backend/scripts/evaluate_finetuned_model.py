"""
Precious Edu LLM — Evaluate Fine-Tuned Model & Comparative Analysis CLI

Runs side-by-side comparative evaluation between Phase 7 Base Model and Phase 8 Fine-Tuned Model
on test perplexity, golden prompt behavioral benchmarks, and regression/catastrophic forgetting prompts.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader

BACKEND_ROOT = Path(__file__).resolve().parent.parent   # .../backend
PROJECT_ROOT = BACKEND_ROOT.parent                       # .../Precious Edu LLM
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.fine_tuning.config import FineTuningConfig
from app.ml.fine_tuning.dataset import ConversationalFineTuningDataset
from app.ml.fine_tuning.evaluator import FineTuningEvaluator, GOLDEN_PROMPTS
from app.ml.model.transformer import PreciousTransformer
from app.ml.model.config import ModelConfig
from app.tokenizer.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[PreciousTransformer, ModelConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    raw_config = ckpt.get("model_config", {})
    if isinstance(raw_config, dict):
        cfg = ModelConfig.from_dict(raw_config)
    else:
        cfg = raw_config

    model = PreciousTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned Model vs Pretrained Base Model.")
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/training/run-train_model/checkpoints/latest.pt"),
        help="Path to Phase 7 pretrained base model checkpoint."
    )
    parser.add_argument(
        "--finetuned_checkpoint",
        type=str,
        default=None,
        help="Path to Phase 8 fine-tuned model checkpoint (e.g. artifacts/fine_tuning/<run_id>/checkpoints/latest.pt)."
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=str(PROJECT_ROOT / "data/training/conversational/test.jsonl"),
        help="Path to test split JSONL file."
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/tokenizer/v1"),
        help="Path to tokenizer directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/fine_tuning/evaluation_results"),
        help="Output directory for reports and comparison files."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.load(Path(args.tokenizer_dir))
    config = FineTuningConfig(max_sequence_length=64)

    # Load Test Dataset
    test_path = Path(args.test_dataset)
    if test_path.exists():
        test_ds = ConversationalFineTuningDataset(
            jsonl_path=str(test_path),
            tokenizer=tokenizer,
            config=config,
        )
        test_loader = DataLoader(test_ds, batch_size=config.micro_batch_size, shuffle=False)
    else:
        test_ds, test_loader = None, None
        logger.warning(f"Test dataset not found at {test_path}")

    # Evaluate Base Model
    base_ckpt_path = Path(args.base_checkpoint)
    if not base_ckpt_path.exists():
        logger.error(f"Base checkpoint not found at {base_ckpt_path}")
        sys.exit(1)

    logger.info(f"Evaluating Phase 7 Base Model from: {base_ckpt_path}")
    base_model, _ = load_model_from_checkpoint(str(base_ckpt_path), device)
    base_evaluator = FineTuningEvaluator(base_model, tokenizer, config, device)

    base_test_loss, base_test_ppl = (None, None)
    if test_loader:
        base_test_loss, base_test_ppl = base_evaluator.evaluate_dataset(test_loader)

    base_behavior = base_evaluator.evaluate_behavior(deterministic=True)
    base_regression = base_evaluator.evaluate_regression()

    # Save baseline outputs
    baseline_file = out_dir / "baseline_outputs.jsonl"
    with open(baseline_file, "w", encoding="utf-8") as f:
        for b in base_behavior:
            f.write(json.dumps(b) + "\n")

    # Resolve Fine-tuned Checkpoint
    finetuned_ckpt_path = args.finetuned_checkpoint
    if not finetuned_ckpt_path:
        # Search for latest fine_tuning run checkpoint
        ft_dir = PROJECT_ROOT / "artifacts/fine_tuning"
        runs = sorted(ft_dir.glob("run-finetune-*"))
        if runs:
            latest_run = runs[-1]
            candidate = latest_run / "checkpoints" / "latest.pt"
            if candidate.exists():
                finetuned_ckpt_path = str(candidate)

    if not finetuned_ckpt_path or not Path(finetuned_ckpt_path).exists():
        logger.warning("No fine-tuned checkpoint found to compare. Reporting baseline metrics only.")
        finetuned_results = None
    else:
        logger.info(f"Evaluating Phase 8 Fine-Tuned Model from: {finetuned_ckpt_path}")
        ft_model, _ = load_model_from_checkpoint(finetuned_ckpt_path, device)
        ft_evaluator = FineTuningEvaluator(ft_model, tokenizer, config, device)

        ft_test_loss, ft_test_ppl = (None, None)
        if test_loader:
            ft_test_loss, ft_test_ppl = ft_evaluator.evaluate_dataset(test_loader)

        ft_behavior = ft_evaluator.evaluate_behavior(deterministic=True)
        ft_regression = ft_evaluator.evaluate_regression()

        finetuned_file = out_dir / "finetuned_outputs.jsonl"
        with open(finetuned_file, "w", encoding="utf-8") as f:
            for ft in ft_behavior:
                f.write(json.dumps(ft) + "\n")

        finetuned_results = {
            "checkpoint": finetuned_ckpt_path,
            "test_loss": ft_test_loss,
            "test_perplexity": ft_test_ppl,
            "behavior": ft_behavior,
            "regression": ft_regression,
        }

    # Summary Report
    report = {
        "dataset_version": "0.1.0",
        "tokenizer_version": tokenizer.config.TOKENIZER_VERSION,
        "base_model": {
            "checkpoint": str(base_ckpt_path),
            "test_loss": base_test_loss,
            "test_perplexity": base_test_ppl,
            "behavior": base_behavior,
            "regression": base_regression,
        },
        "finetuned_model": finetuned_results,
    }

    report_path = out_dir / "final_training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========================================================")
    print("PHASE 8 EVALUATION & BEHAVIORAL COMPARISON REPORT")
    print("========================================================")
    if base_test_loss is not None:
        print(f"Base Model Test Loss       : {base_test_loss:.4f} (Perplexity: {base_test_ppl:.4f})")
    if finetuned_results and finetuned_results["test_loss"] is not None:
        print(f"Fine-Tuned Model Test Loss : {finetuned_results['test_loss']:.4f} (Perplexity: {finetuned_results['test_perplexity']:.4f})")
    print("\nGolden Prompts Comparison:")
    for idx, prompt_info in enumerate(GOLDEN_PROMPTS):
        p_text = prompt_info["prompt"][-1]["content"]
        base_resp = base_behavior[idx]["generated_response"]
        print(f"\nUser: {p_text}")
        print(f"Base Model Response       : {base_resp}")
        if finetuned_results:
            ft_resp = finetuned_results["behavior"][idx]["generated_response"]
            print(f"Fine-Tuned Model Response : {ft_resp}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
