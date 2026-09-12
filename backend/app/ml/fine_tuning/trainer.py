"""
Precious Edu LLM — Fine-Tuning Trainer Engine

Executes causal conversational fine-tuning with assistant-only loss masking,
strict pretrained checkpoint validation, parameter freezing, gradient accumulation,
AdamW optimizer, cosine warmup scheduler, atomic checkpointing, and metrics tracking.
"""

import os
import math
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ml.fine_tuning.config import FineTuningConfig
from app.ml.fine_tuning.exceptions import CheckpointCompatibilityError, FineTuningError
from app.ml.fine_tuning.dataset import ConversationalFineTuningDataset
from app.ml.fine_tuning.evaluator import FineTuningEvaluator
from app.ml.model.transformer import PreciousTransformer
from app.ml.model.config import ModelConfig
from app.ml.training.optimizer import configure_optimizer
from app.ml.training.scheduler import CosineWarmupScheduler
from app.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class FineTuningEngine:
    """
    Engine orchestrating Phase 8 conversational fine-tuning.
    """

    def __init__(
        self,
        config: FineTuningConfig,
        tokenizer: Tokenizer,
        train_dataset: ConversationalFineTuningDataset,
        val_dataset: Optional[ConversationalFineTuningDataset] = None,
        run_id: Optional[str] = None
    ):
        config.validate()
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_id = run_id or f"run-finetune-{timestamp}"

        # Resolve Run Directories
        self.run_dir = Path(self.config.artifacts_dir) / self.run_id
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.logs_dir = self.run_dir / "logs"
        self.final_dir = self.run_dir / "final"

        for d in [self.checkpoint_dir, self.logs_dir, self.final_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.run_dir / "metrics.jsonl"

        # Resolve Device
        self.device = self._resolve_device()

        # Load Pretrained Model & Validate Checkpoint
        self.model, self.model_config = self._load_and_verify_pretrained_model()
        self.model.to(self.device)

        # Apply Optional Parameter Freezing
        self._apply_freezing_strategy()

        # Build DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False,
        ) if self.val_dataset else None

        self.optimizer = configure_optimizer(
            model=self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.max_steps,
            min_learning_rate=self.config.min_learning_rate,
        )

        # Evaluator
        self.evaluator = FineTuningEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
        )

        # Loss Function with ignore_index for Assistant Loss Masking
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)

        # State tracking
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def _resolve_device(self) -> torch.device:
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.config.device == "cpu":
            return torch.device("cpu")
        elif self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def _load_and_verify_pretrained_model(self) -> Tuple[PreciousTransformer, ModelConfig]:
        """
        Loads Phase 7 pretrained checkpoint and verifies strict architecture & tokenizer compatibility.
        """
        ckpt_path = Path(self.config.pretrained_checkpoint_path)
        if not ckpt_path.exists():
            raise CheckpointCompatibilityError(
                f"Pretrained checkpoint not found at: {ckpt_path}. Verify Phase 7 execution."
            )

        logger.info(f"Loading pretrained checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if "model_config" not in checkpoint:
            raise CheckpointCompatibilityError("Checkpoint is missing required 'model_config' metadata.")

        raw_config = checkpoint["model_config"]
        if isinstance(raw_config, dict):
            model_config = ModelConfig.from_dict(raw_config)
        else:
            model_config = raw_config

        # Check vocabulary size compatibility
        if model_config.vocab_size != self.tokenizer.vocab_size:
            raise CheckpointCompatibilityError(
                f"Vocabulary size mismatch: Model config has {model_config.vocab_size}, "
                f"but Tokenizer has {self.tokenizer.vocab_size}."
            )

        model = PreciousTransformer(model_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.info(f"Successfully loaded pretrained model ({model.num_parameters['total']:,} params).")
        return model, model_config

    def _apply_freezing_strategy(self) -> None:
        """Applies configured parameter freezing strategy."""
        strat = self.config.freeze_strategy
        if strat == "none":
            return

        logger.info(f"Applying freeze strategy: {strat}")
        if strat == "embeddings":
            for param in self.model.embedding.parameters():
                param.requires_grad = False

        elif strat == "first_n":
            n = self.config.freeze_first_n_layers
            for i in range(min(n, len(self.model.blocks))):
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = False

        elif strat == "all_except_head":
            for name, param in self.model.named_parameters():
                if "lm_head" not in name:
                    param.requires_grad = False

    def save_checkpoint(self, tag: str) -> Path:
        """Saves atomic checkpoint file."""
        path = self.checkpoint_dir / f"{tag}.pt"
        checkpoint_data = {
            "step": self.current_step,
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_config": self.model_config.to_dict(),
            "finetuning_config": self.config.to_dict(),
            "tokenizer_version": self.tokenizer.config.TOKENIZER_VERSION,
            "best_val_loss": self.best_val_loss,
            "rng_state": torch.get_rng_state(),
        }

        tmp_path = path.with_suffix(".tmp")
        torch.save(checkpoint_data, tmp_path)
        os.replace(tmp_path, path)
        logger.info(f"Saved fine-tuning checkpoint: {path.name} (step {self.current_step})")
        return path

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resumes fine-tuning training from existing checkpoint."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FineTuningError(f"Resume checkpoint file not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_step = checkpoint.get("step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].cpu())

        logger.info(f"Resumed training from checkpoint {ckpt_path.name} at step {self.current_step}")

    def train(self) -> Dict[str, Any]:
        """
        Executes full fine-tuning training loop.
        """
        logger.info(f"Starting Phase 8 Conversational Fine-Tuning run {self.run_id}...")
        start_time = time.time()
        tokens_processed = 0

        self.model.train()
        accum_loss = 0.0

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            for micro_step, (input_ids, labels, attention_mask) in enumerate(self.train_loader):
                if self.current_step >= self.config.max_steps:
                    break

                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)

                tokens_processed += (attention_mask == 1).sum().item()

                logits = self.model(input_ids, attention_mask=attention_mask)

                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)

                loss = self.loss_fn(logits_flat, labels_flat)
                loss_scaled = loss / self.config.gradient_accumulation_steps
                loss_scaled.backward()

                accum_loss += loss.item()

                if (micro_step + 1) % self.config.gradient_accumulation_steps == 0 or (micro_step + 1) == len(self.train_loader):
                    if self.config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.current_step += 1

                    step_loss = accum_loss / self.config.gradient_accumulation_steps
                    accum_loss = 0.0
                    curr_lr = self.optimizer.param_groups[0]["lr"]

                    # Log metrics
                    if self.current_step % self.config.logging_interval == 0:
                        metric_entry = {
                            "step": self.current_step,
                            "epoch": self.current_epoch,
                            "train_loss": round(step_loss, 4),
                            "train_perplexity": round(math.exp(step_loss) if step_loss < 20 else float("inf"), 4),
                            "learning_rate": curr_lr,
                            "tokens_processed": tokens_processed,
                            "timestamp": time.time(),
                        }
                        with open(self.metrics_file, "a", encoding="utf-8") as mf:
                            mf.write(json.dumps(metric_entry) + "\n")

                    # Evaluation
                    if self.val_loader and (self.current_step % self.config.evaluation_interval == 0):
                        val_loss, val_ppl = self.evaluator.evaluate_dataset(self.val_loader)
                        logger.info(
                            f"Step {self.current_step}/{self.config.max_steps} | "
                            f"Train Loss: {step_loss:.4f} | Val Loss: {val_loss:.4f} | "
                            f"Val PPL: {val_ppl:.4f} | LR: {curr_lr:.6f}"
                        )

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best")

                    # Periodic checkpoint
                    if self.current_step % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(f"step-{self.current_step}")

            if self.current_step >= self.config.max_steps:
                break

        # Save latest checkpoint
        self.save_checkpoint("latest")

        # Save final artifacts
        final_model_path = self.final_dir / "model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config.to_dict(),
            "finetuning_config": self.config.to_dict(),
            "tokenizer_version": self.tokenizer.config.TOKENIZER_VERSION,
        }, final_model_path)

        with open(self.final_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        elapsed = time.time() - start_time
        logger.info(f"Fine-tuning complete in {elapsed:.2f}s across {self.current_step} steps.")

        return {
            "run_id": self.run_id,
            "final_step": self.current_step,
            "best_val_loss": self.best_val_loss,
            "elapsed_seconds": elapsed,
            "final_model_path": str(final_model_path),
        }
