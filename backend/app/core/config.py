"""
Precious Edu LLM — Application Configuration

Centralized settings management using pydantic-settings.
All configuration is loaded from environment variables / .env file.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Pydantic-settings automatically reads from .env file and
    environment variables. Environment variables take precedence.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Server ---
    APP_NAME: str = "Precious AI"
    APP_ENV: str = "development"
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    APP_DEBUG: bool = True
    APP_CORS_ORIGINS: str = "http://localhost,http://localhost:80,http://localhost:8080,http://localhost:5173,http://localhost:3000,http://127.0.0.1,http://127.0.0.1:80,http://127.0.0.1:8080,http://127.0.0.1:5173,http://127.0.0.1:3000"

    # --- MongoDB ---
    MONGODB_URI: str = "mongodb://127.0.0.1:27017"
    MONGODB_DB_NAME: str = "precious_edu_llm"


    # --- Model ---
    MODEL_CHECKPOINT_PATH: str = "../checkpoints/model/latest.pt"
    TOKENIZER_PATH: str = "../checkpoints/tokenizer/"

    # --- Rate Limiting ---
    RATE_LIMIT_PER_MINUTE: int = 30

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Generation ---
    MAX_MESSAGE_LENGTH: int = 2000
    MAX_RESPONSE_TOKENS: int = 512
    GENERATION_TEMPERATURE: float = 0.7
    GENERATION_TOP_K: int = 50
    GENERATION_TOP_P: float = 0.9

    # --- Phase 3 Context & Memory ---
    CONTEXT_MAX_MESSAGES: int = 10
    CONTEXT_MAX_CHARACTERS: int = 4000
    MEMORY_MAX_ITEMS: int = 20
    MEMORY_ENABLED: bool = True

    # --- Structured Q&A Knowledge Engine ---
    STRUCTURED_QA_COLLECTION: str = "structured_qa"
    STRUCTURED_QA_MIN_SCORE: float = 20.0
    STRUCTURED_QA_DIRECT_ANSWER_THRESHOLD: float = 45.0
    STRUCTURED_QA_TOP_K: int = 3
    WEBSITE_MIN_SCORE: float = 5.0
    WEBSITE_TOP_K: int = 3


    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.APP_CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.APP_ENV == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.APP_ENV == "production"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses lru_cache to ensure settings are loaded only once.
    This function is used as a FastAPI dependency.
    """
    return Settings()
