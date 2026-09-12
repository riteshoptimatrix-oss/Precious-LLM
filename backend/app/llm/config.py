"""
Precious Edu LLM — LLM Service Configuration

Configuration settings governing model checkpoint resolution, tokenizer paths,
device placement, context budgeting, and default inference generation hyperparameters.
"""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional

from app.ml.fine_tuning.exceptions import FineTuningError


@dataclass
class LLMConfig:
    """
    Configuration settings governing LLM inference service integration.
    """

    # Device & Model Checkpoint
    device: str = "auto"               # "auto", "cpu", "cuda"
    model_checkpoint_path: Optional[str] = None # Auto-resolves latest fine-tuning run if None
    tokenizer_dir: str = "artifacts/tokenizer/v1"

    # Context Bounds (Derived from model max_seq_length)
    max_context_length: int = 64
    max_new_tokens: int = 64

    # Default Hyperparameters for Inference Generation
    temperature: float = 0.65
    top_k: int = 40
    top_p: float = 0.90
    repetition_penalty: float = 1.10
    deterministic: bool = False        # If True, overrides temperature=0.0 and top_k=1

    # Startup Warmup
    warmup_on_startup: bool = True

    def resolve_checkpoint_path(self) -> Path:
        """
        Auto-resolves the latest trained checkpoint: domain-adapted, fine-tuned, or pretrained.
        """
        if self.model_checkpoint_path:
            p = Path(self.model_checkpoint_path)
            if p.exists():
                return p

        candidate_dirs = [
            Path("artifacts/domain_training"),
            Path("backend/artifacts/domain_training"),
            Path("artifacts/fine_tuning"),
            Path("backend/artifacts/fine_tuning"),
        ]

        found_models: list[Path] = []
        for cdir in candidate_dirs:
            if cdir.exists():
                for run_dir in sorted(cdir.glob("run-*")):
                    final_p = run_dir / "final" / "model.pt"
                    if final_p.exists():
                        found_models.append(final_p)
                    best_p = run_dir / "checkpoints" / "best.pt"
                    if best_p.exists():
                        found_models.append(best_p)
                    latest_p = run_dir / "checkpoints" / "latest.pt"
                    if latest_p.exists():
                        found_models.append(latest_p)

        if found_models:
            # Sort by last modification time descending to choose the latest trained checkpoint
            found_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return found_models[0]

        # Fallback to pretrained base model checkpoint
        for pt_path in [
            Path("artifacts/training/run-train_model/checkpoints/best.pt"),
            Path("backend/artifacts/training/run-train_model/checkpoints/best.pt"),
            Path("artifacts/training/run-train_model/checkpoints/latest.pt"),
            Path("backend/artifacts/training/run-train_model/checkpoints/latest.pt"),
        ]:
            if pt_path.exists():
                return pt_path

        raise FileNotFoundError("No valid Phase 8/11 fine-tuned or Phase 7 pretrained checkpoint artifact found.")


    def resolve_tokenizer_dir(self) -> Path:
        """
        Auto-resolves Phase 5 tokenizer artifact directory.
        """
        p = Path(self.tokenizer_dir)
        if p.exists():
            return p

        p_backend = Path("backend") / self.tokenizer_dir
        if p_backend.exists():
            return p_backend

        p_parent = Path("..") / self.tokenizer_dir
        if p_parent.exists():
            return p_parent

        raise FileNotFoundError(f"Tokenizer directory not found at {self.tokenizer_dir}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
