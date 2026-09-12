"""
Precious Edu LLM — Checkpoint Management

Provides atomic saving, loading, validation, and serialization of model checkpoints.
Enforces tokenizer compatibility and configuration integrity.
"""

import os
import json
import hashlib
import tempfile
import datetime
import torch
from typing import Optional, Dict, Any, Tuple

from app.ml.model.config import ModelConfig
from app.ml.model.transformer import PreciousTransformer
from app.ml.model.exceptions import CheckpointError, TokenizerIncompatibilityError


def calculate_sha256(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_checkpoint(
    model: PreciousTransformer,
    save_dir: str,
    tokenizer_version: str = "1.0.0",
    tokenizer_hash: Optional[str] = None,
    training_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save model checkpoint atomically.

    Files saved:
    - config.json: Model architecture configuration
    - model.pt: PyTorch model state dictionary
    - manifest.json: Metadata, parameter count, hashes, tokenizer compatibility info

    Args:
        model: PreciousTransformer instance.
        save_dir: Directory where checkpoint files will be written.
        tokenizer_version: Version of the tokenizer used with this model.
        tokenizer_hash: SHA256 hash of tokenizer vocabulary or manifest.
        training_metadata: Optional training loss or step details.

    Returns:
        Path to save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(save_dir, "config.json")
    model_path = os.path.join(save_dir, "model.pt")
    manifest_path = os.path.join(save_dir, "manifest.json")

    # Write config.json atomically
    temp_config = config_path + ".tmp"
    with open(temp_config, "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, indent=2)
    os.replace(temp_config, config_path)

    # Write model.pt atomically
    temp_model = model_path + ".tmp"
    torch.save(model.state_dict(), temp_model)
    os.replace(temp_model, model_path)

    # Calculate file checksums
    model_sha256 = calculate_sha256(model_path)
    config_sha256 = calculate_sha256(config_path)

    # Create manifest
    manifest_data = {
        "model_version": model.config.model_version,
        "tokenizer_version": tokenizer_version,
        "tokenizer_hash": tokenizer_hash or "",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "vocab_size": model.config.vocab_size,
        "max_seq_length": model.config.max_seq_length,
        "parameters": model.num_parameters,
        "artifacts": {
            "model_pt": {
                "file_name": "model.pt",
                "sha256": model_sha256,
            },
            "config_json": {
                "file_name": "config.json",
                "sha256": config_sha256,
            },
        },
        "training_metadata": training_metadata or {},
    }

    temp_manifest = manifest_path + ".tmp"
    with open(temp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    os.replace(temp_manifest, manifest_path)

    return save_dir


def load_checkpoint(
    save_dir: str,
    expected_tokenizer_version: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[PreciousTransformer, ModelConfig, Dict[str, Any]]:
    """
    Load model checkpoint and validate compatibility.

    Args:
        save_dir: Path to directory containing config.json, model.pt, manifest.json.
        expected_tokenizer_version: Optional tokenizer version to check against.
        device: PyTorch device to load model weights onto.

    Returns:
        Tuple of (PreciousTransformer, ModelConfig, manifest_dict)
    """
    if not os.path.isdir(save_dir):
        raise CheckpointError(f"Checkpoint directory not found: {save_dir}")

    config_path = os.path.join(save_dir, "config.json")
    model_path = os.path.join(save_dir, "model.pt")
    manifest_path = os.path.join(save_dir, "manifest.json")

    for path in (config_path, model_path, manifest_path):
        if not os.path.exists(path):
            raise CheckpointError(f"Missing required checkpoint file: {path}")

    # Load manifest
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise CheckpointError(f"Failed to read manifest.json: {e}")

    # Validate tokenizer compatibility if specified
    if expected_tokenizer_version is not None:
        chk_tok_ver = manifest.get("tokenizer_version", "")
        if chk_tok_ver != expected_tokenizer_version:
            raise TokenizerIncompatibilityError(
                f"Model checkpoint depends on tokenizer version '{chk_tok_ver}', "
                f"which is incompatible with requested tokenizer version '{expected_tokenizer_version}'."
            )

    # Load configuration
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
    except Exception as e:
        raise CheckpointError(f"Failed to read config.json: {e}")

    # Construct model and load state dict
    model = PreciousTransformer(config)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        raise CheckpointError(f"Failed to load model weights from model.pt: {e}")

    if device is not None:
        model.to(device)

    model.eval()
    return model, config, manifest
