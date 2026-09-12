"""
Precious Edu LLM — Dataset Pipeline Configuration

Central settings for dataset processing, quality filters, and deterministic splitting.
"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseSettings):
    """Configuration settings governing dataset processing."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Versions ---
    DATASET_VERSION: str = "0.1.0"
    PIPELINE_VERSION: str = "0.1.0"

    # --- Paths ---
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"

    RAW_DIR: Path = DATA_DIR / "raw"
    INTERMEDIATE_DIR: Path = DATA_DIR / "intermediate"
    QUARANTINE_DIR: Path = INTERMEDIATE_DIR / "quarantine"
    CLEANED_DIR: Path = DATA_DIR / "cleaned"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    TRAINING_DIR: Path = DATA_DIR / "training"
    EVALUATION_DIR: Path = DATA_DIR / "evaluation"
    METADATA_DIR: Path = DATA_DIR / "metadata"

    # --- Filtering & Budgeting ---
    MIN_TEXT_CHARACTERS: int = 2
    MAX_TEXT_CHARACTERS: int = 10000
    MAX_REPETITION_RATIO: float = 0.5

    # --- Splitting Ratios & Seed ---
    TRAIN_RATIO: float = 0.90
    VALIDATION_RATIO: float = 0.05
    TEST_RATIO: float = 0.05
    SPLIT_SEED: int = 42

    # --- Pipeline Feature Flags ---
    ENABLE_DEDUPLICATION: bool = True
    ENABLE_QUALITY_FILTERS: bool = True
    STRICT_MODE: bool = False

    def ensure_directories(self) -> None:
        """Create all required data subdirectories if they do not exist."""
        for path in [
            self.RAW_DIR / "text",
            self.RAW_DIR / "conversations",
            self.RAW_DIR / "external",
            self.QUARANTINE_DIR,
            self.CLEANED_DIR,
            self.PROCESSED_DIR,
            self.TRAINING_DIR,
            self.EVALUATION_DIR,
            self.METADATA_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_dataset_config() -> DatasetConfig:
    """Get cached DatasetConfig settings."""
    cfg = DatasetConfig()
    cfg.ensure_directories()
    return cfg
