"""
Precious Edu LLM — Pretraining Engine

Main training engine orchestrating causal LM pretraining, gradient accumulation,
numerical stability checks, gradient clipping, evaluation, metrics logging, checkpointing,
and resume operations.
"""

import os
import sys
import math
import time
import json
import signal
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ml.model import PreciousTransformer, ModelConfig
from app.ml.training.config import TrainingConfig
from app.ml.training.exceptions import TrainingError, NumericalInstabilityError
from app.ml.training.reproducibility import set_seed
from app.ml.training.optimizer import configure_optimizer
from app.ml.training.scheduler import CosineWarmupScheduler
from app.ml.training.evaluator import Evaluator
from app.ml.training.metrics import MetricsTracker
from app.ml.training.checkpointing import save_training_checkpoint, load_training_checkpoint


class PretrainingEngine:
    """
    Core LLM Pretraining Engine.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: PreciousTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: Path,
    ):
        config.validate()
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Device selection
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)
        self.evaluator = Evaluator(self.model, self.val_loader, self.device)
        self.metrics = MetricsTracker(self.run_dir)

    def dry_run(self) -> Dict[str, Any]:
        """
        Execute pre-flight validation check without running full training.
        """
        print("\n========================================")
        print("PRECIOUS AI PRETRAINING PRE-FLIGHT CHECK")
        print("========================================")

        set_seed(self.config.seed)

        print(f"Device               : {self.device}")
        print(f"CUDA Available       : {torch.cuda.is_available()}")
        print(f"PyTorch Version      : {torch.__version__}")
        print(f"Model Version        : {self.model.config.model_version}")
        print(f"Vocab Size           : {self.model.config.vocab_size}")
        print(f"Context Length       : {self.model.config.max_seq_length}")
        print(f"Model Parameters     : {self.model.num_parameters['total']:,}")
        print(f"Train Dataset Samples: {len(self.train_loader.dataset):,}")
        print(f"Val Dataset Samples  : {len(self.val_loader.dataset):,}")
        print(f"Batch Size           : {self.config.batch_size}")
        print(f"Micro Batch Size     : {self.config.micro_batch_size}")
        print(f"Grad Accumulation    : {self.config.gradient_accumulation_steps}")
        print(f"Learning Rate        : {self.config.learning_rate}")
        print(f"Run Directory        : {self.run_dir}")

        # Perform a single forward/backward dry-run step
        self.model.train()
        input_ids, target_ids = next(iter(self.train_loader))
        input_ids = input_ids[: self.config.micro_batch_size].to(self.device)
        target_ids = target_ids[: self.config.micro_batch_size].to(self.device)

        optimizer = configure_optimizer(self.model, self.config.learning_rate)
        optimizer.zero_grad()

        logits, loss = self.model(input_ids, targets=target_ids)
        loss.backward()

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if p.grad is None or torch.isnan(p.grad).any():
                    raise NumericalInstabilityError(f"Dry-run gradient check failed on {name}")

        optimizer.step()

        print("Dry-run verification completed successfully.")
        print("========================================\n")

        return {
            "status": "PASS",
            "device": str(self.device),
            "model_parameters": self.model.num_parameters["total"],
            "dry_run_loss": round(loss.item(), 4),
        }

    def train(self, resume_checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute pretraining pipeline.

        Args:
            resume_checkpoint_path: Optional path to checkpoint .pt file to resume from.

        Returns:
            Dictionary containing final pretraining summary and metrics.
        """
        set_seed(self.config.seed)

        # Configure optimizer & scheduler
        optimizer = configure_optimizer(
            self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.total_steps,
            min_learning_rate=self.config.min_learning_rate,
        )

        start_step = 0
        start_epoch = 0
        best_val_loss = float("inf")
        best_val_ppl = float("inf")

        # Resume if requested
        if resume_checkpoint_path:
            print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
            start_step, start_epoch, last_loss, last_metrics, _, _ = load_training_checkpoint(
                resume_checkpoint_path,
                self.model,
                optimizer,
                scheduler,
                device=self.device,
            )
            scheduler.total_steps = self.config.total_steps
            best_val_loss = last_metrics.get("val_loss", float("inf"))
            print(f"Resumed at step {start_step}, epoch {start_epoch}, last loss {last_loss:.4f}")

        # Save initial configuration snapshot & run manifest
        self._write_run_manifest(status="RUNNING")

        print("\n========================================")
        print("PRECIOUS AI LLM PRETRAINING STARTED")
        print("========================================")

        self.model.train()
        step = start_step
        epoch = start_epoch
        accumulated_loss = 0.0
        micro_step = 0

        stop_requested = False

        def handle_interrupt(sig, frame):
            nonlocal stop_requested
            print("\nInterrupt signal received. Saving emergency checkpoint...")
            stop_requested = True

        signal.signal(signal.SIGINT, handle_interrupt)

        start_train_time = time.time()
        initial_train_loss = None

        # Calculate micro-batches to skip if resuming
        micro_batches_to_skip = start_step * self.config.gradient_accumulation_steps
        micro_batches_skipped = 0

        epoch = start_epoch - 1 if start_epoch > 0 else 0

        while step < self.config.total_steps and epoch < self.config.epochs and not stop_requested:
            epoch += 1
            for input_ids, target_ids in self.train_loader:
                if micro_batches_skipped < micro_batches_to_skip:
                    micro_batches_skipped += 1
                    continue

                if step >= self.config.total_steps or stop_requested:
                    break

                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                # Forward pass & loss scaling for accumulation
                logits, loss = self.model(input_ids, targets=target_ids)

                if initial_train_loss is None:
                    initial_train_loss = loss.item()

                if torch.isnan(loss) or torch.isinf(loss):
                    raise NumericalInstabilityError(f"NaN or Inf loss detected at step {step + 1}: {loss.item()}")

                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                accumulated_loss += loss.item()
                micro_step += 1

                # Step boundary after gradient accumulation
                if micro_step % self.config.gradient_accumulation_steps == 0:
                    step += 1

                    # Check gradient health & clip
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    ).item()

                    if math.isnan(grad_norm) or math.isinf(grad_norm):
                        raise NumericalInstabilityError(f"NaN or Inf gradient norm detected at step {step}")

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    mean_step_loss = accumulated_loss / self.config.gradient_accumulation_steps
                    accumulated_loss = 0.0

                    # Compute training perplexity
                    try:
                        train_ppl = math.exp(mean_step_loss)
                        if math.isinf(train_ppl) or train_ppl > 1e8:
                            train_ppl = 1e8
                    except OverflowError:
                        train_ppl = 1e8

                    current_lr = scheduler.get_lr()[0]
                    tokens_in_step = input_ids.size(0) * input_ids.size(1) * self.config.gradient_accumulation_steps

                    # Evaluate validation dataset if interval reached
                    val_loss, val_ppl = None, None
                    if step % self.config.evaluation_interval == 0 or step == self.config.total_steps:
                        val_loss, val_ppl = self.evaluator.evaluate(max_batches=20)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_val_ppl = val_ppl
                            save_training_checkpoint(
                                self.model,
                                optimizer,
                                scheduler,
                                self.config,
                                step,
                                epoch,
                                mean_step_loss,
                                {"val_loss": val_loss, "val_ppl": val_ppl},
                                str(self.run_dir),
                                is_best=True,
                            )

                    # Log metrics
                    if step % self.config.logging_interval == 0 or step == self.config.total_steps or val_loss is not None:
                        self.metrics.log(
                            step=step,
                            epoch=epoch,
                            train_loss=mean_step_loss,
                            train_ppl=train_ppl,
                            lr=current_lr,
                            grad_norm=grad_norm,
                            tokens_in_step=tokens_in_step,
                            val_loss=val_loss,
                            val_ppl=val_ppl,
                        )

                    # Checkpoint periodically
                    if step % self.config.checkpoint_interval == 0 or step == self.config.total_steps:
                        save_training_checkpoint(
                            self.model,
                            optimizer,
                            scheduler,
                            self.config,
                            step,
                            epoch,
                            mean_step_loss,
                            {"val_loss": val_loss or best_val_loss},
                            str(self.run_dir),
                            is_best=False,
                        )

        total_train_time = time.time() - start_train_time
        final_val_loss, final_val_ppl = self.evaluator.evaluate()

        # Save final checkpoint
        final_ckpt_path = save_training_checkpoint(
            self.model,
            optimizer,
            scheduler,
            self.config,
            step,
            epoch,
            mean_step_loss if 'mean_step_loss' in locals() else 0.0,
            {"val_loss": final_val_loss, "val_ppl": final_val_ppl},
            str(self.run_dir),
            is_best=False,
        )

        status_str = "INTERRUPTED" if stop_requested else "COMPLETED"
        summary = {
            "status": status_str,
            "run_id": self.run_dir.name,
            "total_steps": step,
            "completed_epochs": epoch,
            "tokens_processed": self.metrics.tokens_seen,
            "initial_train_loss": round(initial_train_loss or 0.0, 4),
            "final_train_loss": round(mean_step_loss if 'mean_step_loss' in locals() else 0.0, 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_ppl": round(best_val_ppl, 2),
            "final_val_loss": round(final_val_loss, 4),
            "final_val_ppl": round(final_val_ppl, 2),
            "total_train_time_sec": round(total_train_time, 2),
            "avg_tokens_per_sec": round(self.metrics.tokens_seen / max(0.001, total_train_time), 1),
            "checkpoint_path": final_ckpt_path,
        }

        self._write_run_manifest(status=status_str, summary=summary)

        print("\n========================================")
        print(f"PRETRAINING {status_str}")
        print(f"Total Steps       : {step}/{self.config.total_steps}")
        print(f"Tokens Processed  : {self.metrics.tokens_seen:,}")
        print(f"Final Train Loss  : {summary['final_train_loss']}")
        print(f"Best Val Loss     : {summary['best_val_loss']}")
        print(f"Best Val PPL      : {summary['best_val_ppl']}")
        print("========================================\n")

        return summary

    def _write_run_manifest(self, status: str = "RUNNING", summary: Optional[Dict] = None) -> None:
        """Write run_manifest.json to run directory."""
        manifest_data = {
            "run_id": self.run_dir.name,
            "status": status,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "device": str(self.device),
            "model_version": self.model.config.model_version,
            "tokenizer_version": self.config.dataset_dir,
            "training_config": self.config.to_dict(),
            "model_config": self.model.config.to_dict(),
            "summary": summary or {},
        }
        with open(self.run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
