"""
Precious Edu LLM — Pretraining Checkpoint Manager

Saves and loads full pretraining checkpoints including model state, optimizer state,
scheduler state, step count, epoch, loss, and configuration metadata.
Supports atomic writes and resuming.
"""

import os
import json
import shutil
import datetime
import torch
from typing import Dict, Any, Tuple, Optional

from app.ml.model import PreciousTransformer, ModelConfig
from app.ml.training.config import TrainingConfig
from app.ml.training.exceptions import TrainingError


def save_training_checkpoint(
    model: PreciousTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: TrainingConfig,
    step: int,
    epoch: int,
    loss: float,
    metrics: Dict[str, Any],
    save_dir: str,
    is_best: bool = False,
    scaler: Optional[Any] = None,
) -> str:
    """
    Save training checkpoint atomically.

    Args:
        model: PreciousTransformer model.
        optimizer: AdamW optimizer.
        scheduler: CosineWarmupScheduler.
        config: TrainingConfig instance.
        step: Current step index.
        epoch: Current epoch index.
        loss: Current loss.
        metrics: Dictionary of current metrics.
        save_dir: Run directory.
        is_best: If True, copies checkpoint to best.pt.
        scaler: Optional GradScaler if mixed precision FP16 used.

    Returns:
        Path to saved checkpoint file.
    """
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    checkpoint_data = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else {},
        "scaler_state_dict": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "model_config": model.config.to_dict(),
        "training_config": config.to_dict(),
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Atomic write to latest.pt
    latest_path = os.path.join(checkpoints_dir, "latest.pt")
    temp_path = latest_path + ".tmp"
    torch.save(checkpoint_data, temp_path)
    os.replace(temp_path, latest_path)

    # Save step checkpoint (e.g. step-100.pt)
    step_path = os.path.join(checkpoints_dir, f"step-{step}.pt")
    shutil.copyfile(latest_path, step_path)

    # Copy to best.pt if is_best
    if is_best:
        best_path = os.path.join(checkpoints_dir, "best.pt")
        shutil.copyfile(latest_path, best_path)

    return latest_path


def load_training_checkpoint(
    checkpoint_path: str,
    model: PreciousTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, int, float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load training checkpoint and restore states.

    Args:
        checkpoint_path: Path to checkpoint file (.pt).
        model: PreciousTransformer model instance.
        optimizer: Optional AdamW optimizer instance.
        scheduler: Optional CosineWarmupScheduler instance.
        scaler: Optional GradScaler.
        device: PyTorch device.

    Returns:
        Tuple of (step, epoch, loss, metrics, model_config_dict, training_config_dict)
    """
    if not os.path.exists(checkpoint_path):
        raise TrainingError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    except Exception as e:
        raise TrainingError(f"Failed to load checkpoint file {checkpoint_path}: {e}")

    # Verify model state dict compatibility
    try:
        model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    except Exception as e:
        raise TrainingError(f"Model state dict loading failed: {e}")

    # Restore optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    # Restore scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint_data:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

    # Restore scaler state
    if scaler is not None and checkpoint_data.get("scaler_state_dict") is not None:
        if hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(checkpoint_data["scaler_state_dict"])

    step = checkpoint_data.get("step", 0)
    epoch = checkpoint_data.get("epoch", 0)
    loss = checkpoint_data.get("loss", 0.0)
    metrics = checkpoint_data.get("metrics", {})
    model_config = checkpoint_data.get("model_config", {})
    training_config = checkpoint_data.get("training_config", {})

    return step, epoch, loss, metrics, model_config, training_config
