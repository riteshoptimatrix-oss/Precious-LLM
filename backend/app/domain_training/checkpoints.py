"""
Precious Edu LLM — Domain Checkpoint Manager

Saves and loads Phase 11 domain fine-tuning checkpoints and metadata manifests.
Supports training resume and model versioning.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch

from app.domain_training.config import DomainTrainingConfig

logger = logging.getLogger(__name__)


class DomainCheckpointManager:
    """
    Manages checkpoint saving, metadata serialization, and resume loading.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.final_dir = self.run_dir / "final"
        self.logs_dir = self.run_dir / "logs"

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        epoch: int,
        metrics: Dict[str, Any],
        config: DomainTrainingConfig,
        name: str = "latest"
    ) -> Path:
        """
        Saves a model checkpoint dictionary.
        """
        ckpt_path = self.checkpoints_dir / f"{name}.pt"

        # Model configuration metadata
        model_config = getattr(model, "config", None)
        model_config_dict = model_config.__dict__ if model_config and hasattr(model_config, "__dict__") else {}

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "base_checkpoint_path": str(config.resolved_base_checkpoint),
            "tokenizer_dir": str(config.resolved_tokenizer_dir),
            "model_config": model_config_dict,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        torch.save(ckpt, ckpt_path)
        logger.info(f"Saved domain checkpoint to {ckpt_path}")
        return ckpt_path

    def save_final(
        self,
        model: torch.nn.Module,
        config: DomainTrainingConfig,
        final_metrics: Dict[str, Any]
    ) -> Path:
        """
        Saves final production model artifact to final/model.pt.
        """
        final_path = self.final_dir / "model.pt"

        model_config = getattr(model, "config", None)
        model_config_dict = model_config.__dict__ if model_config and hasattr(model_config, "__dict__") else {}

        ckpt = {
            "model_state_dict": model.state_dict(),
            "metrics": final_metrics,
            "base_checkpoint_path": str(config.resolved_base_checkpoint),
            "tokenizer_dir": str(config.resolved_tokenizer_dir),
            "model_config": model_config_dict,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        torch.save(ckpt, final_path)

        # Save manifest.json
        manifest = {
            "version": "1.0.0",
            "phase": "Phase 11 — Domain Fine-Tuning",
            "base_checkpoint": str(config.resolved_base_checkpoint),
            "final_model_path": str(final_path),
            "tokenizer_dir": str(config.resolved_tokenizer_dir),
            "final_metrics": final_metrics,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved final domain model to {final_path}")
        return final_path

    def load_checkpoint(
        self,
        ckpt_path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Tuple[int, int, Dict[str, Any]]:
        """
        Loads checkpoint weights and state for resuming training.
        """
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        ckpt = torch.load(path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        step = ckpt.get("step", 0)
        epoch = ckpt.get("epoch", 0)
        metrics = ckpt.get("metrics", {})

        logger.info(f"Loaded domain checkpoint from {path} (Step {step}, Epoch {epoch})")
        return step, epoch, metrics
