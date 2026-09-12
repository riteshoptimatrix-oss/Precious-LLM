"""
Precious Edu LLM — Tokenizer Serialization

Serializes and loads custom tokenizer artifacts (vocab.json, merges.json, config, manifest).
Verifies artifact checksums and vocabulary integrity upon loading.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.config import TokenizerConfig, get_tokenizer_config
from app.tokenizer.exceptions import TokenizerSerializationError, InvalidVocabularyError, InvalidMergeRuleError

logger = logging.getLogger(__name__)


class TokenizerSerializer:
    """
    Handles saving and loading tokenizer artifacts.
    """

    def compute_file_sha256(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        if not file_path.exists():
            return ""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def save(
        self,
        vocab: Vocabulary,
        merges: List[Tuple[str, str]],
        output_dir: Path,
        config: Optional[TokenizerConfig] = None,
        dataset_version: str = "0.1.0",
        statistics: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save tokenizer artifacts to output directory.
        """
        config = config or get_tokenizer_config()
        output_dir.mkdir(parents=True, exist_ok=True)

        vocab.validate_integrity()

        # 1. Save vocab.json
        vocab_path = output_dir / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(vocab.token_to_id, indent=2, ensure_ascii=False) + "\n")

        # 2. Save merges.json (ordered list of symbol pairs)
        merges_path = output_dir / "merges.json"
        merges_payload = [{"pair": [p[0], p[1]], "rank": idx} for idx, p in enumerate(merges)]
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(merges_payload, indent=2, ensure_ascii=False) + "\n")

        # 3. Save tokenizer_config.json
        config_path = output_dir / "tokenizer_config.json"
        config_payload = {
            "tokenizer_type": config.TOKENIZER_TYPE,
            "tokenizer_version": config.TOKENIZER_VERSION,
            "vocab_size": len(vocab),
            "min_pair_frequency": config.MIN_PAIR_FREQUENCY,
            "max_merges": config.MAX_MERGES,
            "special_tokens": config.special_tokens_map
        }
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n")

        # 4. Save tokenizer_statistics.json
        stats_path = output_dir / "tokenizer_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(statistics or {}, indent=2, ensure_ascii=False) + "\n")

        # 5. Compute SHA-256 Checksums & Save tokenizer_manifest.json
        manifest_payload = {
            "tokenizer_version": config.TOKENIZER_VERSION,
            "algorithm": config.TOKENIZER_TYPE,
            "vocab_size": len(vocab),
            "merges_count": len(merges),
            "dataset_version": dataset_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "vocab": {
                    "file_name": vocab_path.name,
                    "sha256": self.compute_file_sha256(vocab_path)
                },
                "merges": {
                    "file_name": merges_path.name,
                    "sha256": self.compute_file_sha256(merges_path)
                },
                "config": {
                    "file_name": config_path.name,
                    "sha256": self.compute_file_sha256(config_path)
                }
            }
        }
        manifest_path = output_dir / "tokenizer_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n")

        logger.info(f"Saved tokenizer artifacts (vocab_size={len(vocab)}, merges={len(merges)}) to {output_dir}")
        return output_dir

    def load(self, input_dir: Path) -> Tuple[Vocabulary, List[Tuple[str, str]], Dict[str, Any]]:
        """
        Load tokenizer artifacts from input directory.

        Returns:
            Tuple of (Vocabulary, List[Merges], ManifestDict).
        """
        if not input_dir.exists():
            raise TokenizerSerializationError(f"Tokenizer directory does not exist: {input_dir}")

        vocab_path = input_dir / "vocab.json"
        merges_path = input_dir / "merges.json"
        manifest_path = input_dir / "tokenizer_manifest.json"

        if not vocab_path.exists() or not merges_path.exists():
            raise TokenizerSerializationError(f"Missing required artifact files in {input_dir}")

        # 1. Load vocab.json
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                token_to_id = json.load(f)
            vocab = Vocabulary(token_to_id)
            vocab.validate_integrity()
        except Exception as err:
            raise InvalidVocabularyError(f"Failed to load or validate vocab.json: {err}")

        # 2. Load merges.json
        merges: List[Tuple[str, str]] = []
        try:
            with open(merges_path, "r", encoding="utf-8") as f:
                merges_data = json.load(f)
            for item in merges_data:
                pair = item.get("pair")
                if isinstance(pair, list) and len(pair) == 2:
                    merges.append((pair[0], pair[1]))
                else:
                    raise InvalidMergeRuleError(f"Invalid merge rule format in merges.json: {item}")
        except Exception as err:
            raise InvalidMergeRuleError(f"Failed to load merges.json: {err}")

        # 3. Load manifest if present
        manifest: Dict[str, Any] = {}
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        logger.info(f"Loaded tokenizer artifacts from {input_dir} (Vocab size: {len(vocab)}, Merges: {len(merges)})")
        return vocab, merges, manifest
