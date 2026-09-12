"""
Precious Edu LLM — Training Metrics Logger

Tracks, formats, logs, and persists structured training & validation metrics.
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional


class MetricsTracker:
    """
    Logs structured training metrics to console and appends JSON lines to metrics.jsonl.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.start_time = time.time()
        self.tokens_seen = 0

    def log(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        train_ppl: float,
        lr: float,
        grad_norm: float,
        tokens_in_step: int,
        val_loss: Optional[float] = None,
        val_ppl: Optional[float] = None,
        print_console: bool = True,
    ) -> Dict[str, Any]:
        """
        Record and log metrics for current step.
        """
        self.tokens_seen += tokens_in_step
        elapsed = time.time() - self.start_time
        tokens_per_sec = self.tokens_seen / max(0.001, elapsed)

        entry = {
            "step": step,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_ppl": round(train_ppl, 2),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "val_ppl": round(val_ppl, 2) if val_ppl is not None else None,
            "lr": f"{lr:.6e}",
            "grad_norm": round(grad_norm, 4),
            "tokens_seen": self.tokens_seen,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "elapsed_sec": round(elapsed, 2),
        }

        # Append to JSONL file
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if print_console:
            val_str = f" | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}" if val_loss is not None else ""
            print(
                f"[Step {step:5d}] Loss: {train_loss:.4f} | PPL: {train_ppl:.2f} | "
                f"LR: {lr:.2e} | GradNorm: {grad_norm:.2f} | Tokens: {self.tokens_seen:,}{val_str}"
            )

        return entry
