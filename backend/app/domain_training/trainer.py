"""
Precious Edu LLM — Domain Fine-Tuning Trainer

Executes supervised domain fine-tuning starting from Phase 8 conversational model.
Uses mixed sampling, assistant-only loss masking, gradient clipping,
reproducible seed management, and checkpointing.
"""

import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.domain_training.checkpoints import DomainCheckpointManager
from app.domain_training.config import DomainTrainingConfig
from app.domain_training.dataset import DomainTrainingDataset
from app.domain_training.sampler import MixedDataset
from app.ml.model.config import ModelConfig
from app.ml.model.transformer import PreciousTransformer
from app.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class DomainTrainer:
    """
    Supervised domain fine-tuning trainer.
    """

    def __init__(self, config: DomainTrainingConfig):
        self.config = config
        self._set_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu")
        logger.info(f"Using device for Domain Fine-Tuning: {self.device}")

        # Load Tokenizer
        tok_path = config.resolved_tokenizer_dir
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer directory not found at: {tok_path}")
        self.tokenizer = Tokenizer.load(tok_path)

        # Load Base Phase 8 Model
        base_ckpt = config.resolved_base_checkpoint
        if not base_ckpt.exists():
            raise FileNotFoundError(f"Base Phase 8 checkpoint not found: {base_ckpt}")

        logger.info(f"Loading base checkpoint from {base_ckpt}")
        checkpoint = torch.load(base_ckpt, map_location="cpu")
        
        if "model_config" in checkpoint:
            raw_config = checkpoint["model_config"]
            self.model_config = ModelConfig.from_dict(raw_config) if isinstance(raw_config, dict) else raw_config
        else:
            self.model_config = ModelConfig()

        self.model = PreciousTransformer(self.model_config)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_tiny_overfit_test(
        self,
        dataset: Optional[DomainTrainingDataset] = None,
        num_steps: int = 15
    ) -> Dict[str, Any]:
        """
        Validates pipeline correctness by overfitting a single mini-batch.
        Returns dict containing pass status and loss metrics.
        """
        logger.info("Running Tiny Overfit Test...")
        if dataset is None:
            train_path = self.config.resolved_data_root / "training" / "train.jsonl"
            if not train_path.exists():
                from app.domain.dataset import DomainDatasetPipeline
                pipeline = DomainDatasetPipeline(data_root=self.config.resolved_data_root)
                pipeline.run_pipeline()
            dataset = DomainTrainingDataset(
                data_path=train_path,
                tokenizer=self.tokenizer,
                max_sequence_length=self.config.max_sequence_length
            )

        if len(dataset) == 0:
            logger.warning("Dataset empty, skipping overfit test.")
            return {"passed": False, "initial_loss": 0.0, "final_loss": 0.0, "loss_reduction": 0.0}

        batch = dataset[0]
        input_ids = batch["input_ids"].unsqueeze(0).to(self.device)
        labels = batch["labels"].unsqueeze(0).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.model.train()

        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()
            logits = self.model(input_ids)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_dropped = final_loss < initial_loss
        loss_red = initial_loss - final_loss
        logger.info(f"Tiny Overfit Test: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}, passed={loss_dropped}")
        return {
            "passed": loss_dropped,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": loss_red
        }

    def run_overfit_test(
        self,
        dataset: Optional[DomainTrainingDataset] = None,
        num_steps: int = 15
    ) -> Dict[str, Any]:
        return self.run_tiny_overfit_test(dataset=dataset, num_steps=num_steps)

    def train(
        self,
        train_dataset: Optional[DomainTrainingDataset] = None,
        val_dataset: Optional[DomainTrainingDataset] = None,
        resume_checkpoint: Optional[str] = None
    ) -> Path:
        """
        Executes full supervised domain fine-tuning loop.
        Returns Path to final model file.
        """
        if train_dataset is None:
            train_path = self.config.resolved_data_root / "training" / "train.jsonl"
            if not train_path.exists():
                from app.domain.dataset import DomainDatasetPipeline
                pipeline = DomainDatasetPipeline(data_root=self.config.resolved_data_root)
                pipeline.run_pipeline()
            train_dataset = DomainTrainingDataset(
                data_path=train_path,
                tokenizer=self.tokenizer,
                max_sequence_length=self.config.max_sequence_length
            )

        if val_dataset is None:
            val_path = self.config.resolved_data_root / "training" / "val.jsonl"
            if val_path.exists():
                val_dataset = DomainTrainingDataset(
                    data_path=val_path,
                    tokenizer=self.tokenizer,
                    max_sequence_length=self.config.max_sequence_length
                )
        """
        Executes full supervised domain fine-tuning loop.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.config.get_run_dir(timestamp_str)
        ckpt_manager = DomainCheckpointManager(run_dir)

        # Save config.json
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            f.write(self.config.model_dump_json(indent=2))

        mixed_train = MixedDataset(
            domain_dataset=train_dataset,
            domain_ratio=self.config.domain_ratio,
            general_ratio=self.config.general_conversation_ratio,
            instruction_ratio=self.config.instruction_ratio,
            seed=self.config.seed
        )

        train_loader = DataLoader(
            mixed_train,
            batch_size=self.config.micro_batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=False
        ) if val_dataset and len(val_dataset) > 0 else None

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay
        )

        total_steps = len(train_loader) * self.config.epochs // self.config.gradient_accumulation_steps
        total_steps = max(1, total_steps)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=self.config.min_learning_rate
        )

        start_step = 0
        start_epoch = 0
        best_val_loss = float("inf")

        if resume_checkpoint:
            start_step, start_epoch, _ = ckpt_manager.load_checkpoint(
                resume_checkpoint, self.model, optimizer, scheduler, device=str(self.device)
            )

        logger.info(f"Starting Domain Fine-Tuning Run: {run_dir.name} ({self.config.epochs} epochs, {len(train_loader)} batches/epoch)")

        metrics_log_path = run_dir / "logs" / "metrics.jsonl"
        metrics_file = open(metrics_log_path, "a", encoding="utf-8")

        global_step = start_step
        start_time = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            self.model.train()
            accum_loss = 0.0

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                if (labels != -100).sum() == 0:
                    continue

                logits = self.model(input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                if torch.isnan(loss):
                    continue

                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                accum_loss += loss.item() * self.config.gradient_accumulation_steps

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    avg_train_loss = accum_loss / self.config.gradient_accumulation_steps
                    accum_loss = 0.0

                    # Compute Perplexity
                    train_ppl = math.exp(min(avg_train_loss, 20.0))

                    metrics = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "train_perplexity": train_ppl,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_sec": time.time() - start_time
                    }

                    metrics_file.write(json.dumps(metrics) + "\n")
                    metrics_file.flush()

                    if global_step % self.config.log_interval == 0:
                        logger.info(f"Epoch {epoch+1}/{self.config.epochs} | Step {global_step} | Loss: {avg_train_loss:.4f} | PPL: {train_ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")

                    # Save Latest Checkpoint
                    if global_step % self.config.checkpoint_interval == 0:
                        ckpt_manager.save_checkpoint(self.model, optimizer, scheduler, global_step, epoch+1, metrics, self.config, name="latest")

            # Epoch Validation Pass
            if val_loader:
                val_loss, val_ppl = self._evaluate(val_loader)
                logger.info(f"Epoch {epoch+1} Evaluation -> Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

                val_metrics = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_perplexity": val_ppl,
                }
                metrics_file.write(json.dumps(val_metrics) + "\n")
                metrics_file.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_manager.save_checkpoint(self.model, optimizer, scheduler, global_step, epoch+1, val_metrics, self.config, name="best")

        metrics_file.close()

        # Save Final Production Model
        final_metrics = {
            "total_steps": global_step,
            "total_epochs": self.config.epochs,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
            "total_time_sec": time.time() - start_time
        }
        final_path = ckpt_manager.save_final(self.model, self.config, final_metrics)

        return {
            "run_dir": str(run_dir),
            "final_model_path": str(final_path),
            "metrics": final_metrics
        }

    def _evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Runs evaluation pass on validation loader."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                if (labels != -100).sum() == 0:
                    continue
                logits = self.model(input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        ppl = math.exp(min(avg_loss, 20.0))
        return avg_loss, ppl
