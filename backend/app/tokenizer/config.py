"""
Precious Edu LLM — Tokenizer Configuration

Central configuration settings and special token definitions for the custom BPE tokenizer.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class TokenizerConfig(BaseSettings):
    """Configuration settings governing tokenizer training, serialization, and vocabulary."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Versions & Metadata ---
    TOKENIZER_TYPE: str = "bpe"
    TOKENIZER_VERSION: str = "1.0.0"

    # --- Vocabulary Boundaries ---
    VOCAB_SIZE: int = 8000
    MIN_PAIR_FREQUENCY: int = 2
    MAX_MERGES: int = 10000

    # --- Special Tokens & Stable IDs ---
    PAD_TOKEN: str = "<pad>"
    UNK_TOKEN: str = "<unk>"
    BOS_TOKEN: str = "<bos>"
    EOS_TOKEN: str = "<eos>"
    SYSTEM_TOKEN: str = "<system>"
    USER_TOKEN: str = "<user>"
    ASSISTANT_TOKEN: str = "<assistant>"

    # --- Directory Paths ---
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts" / "tokenizer" / "v1"

    @property
    def special_tokens_map(self) -> Dict[str, int]:
        """Stable mapping of special tokens to deterministic integer IDs."""
        return {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
            self.SYSTEM_TOKEN: 4,
            self.USER_TOKEN: 5,
            self.ASSISTANT_TOKEN: 6,
        }

    @property
    def special_tokens_list(self) -> List[str]:
        """Ordered list of special token strings."""
        return [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.SYSTEM_TOKEN,
            self.USER_TOKEN,
            self.ASSISTANT_TOKEN,
        ]

    def ensure_artifacts_dir(self) -> Path:
        """Ensure artifacts directory exists."""
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        return self.ARTIFACTS_DIR


@lru_cache()
def get_tokenizer_config() -> TokenizerConfig:
    """Get cached TokenizerConfig settings."""
    cfg = TokenizerConfig()
    cfg.ensure_artifacts_dir()
    return cfg
