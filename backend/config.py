"""
Configuration management using environment variables and .env file.

This module provides a Settings class that loads configuration from
environment variables and .env file using pydantic-settings.
"""
import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Qdrant Cloud Configuration
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "book_content"

    # Cohere API (Embeddings)
    cohere_api_key: str
    cohere_model: str = "embed-multilingual-v3.0"

    # OpenAI/OpenRouter API (Chat Generation)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0

    # OpenRouter Configuration (optional, falls back to OpenAI if not set)
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    router_base_url: Optional[str] = None

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Session Configuration
    max_session_messages: int = 10
    session_timeout_minutes: int = 30

    # Retrieval Configuration
    default_top_k: int = 5
    max_top_k: int = 10
    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 100
    embedding_batch_size: int = 96

    # Ingestion Configuration
    ingest_input_path: str = "../book-content"
    ingest_collection_name: str = "book_content"

    class Config(SettingsConfigDict):
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """
    Get application settings instance.

    Returns:
        Settings: Configured application settings

    Example:
        >>> settings = get_settings()
        >>> print(settings.qdrant_url)
        'https://cluster.qdrant.io'
    """
    return Settings()


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure application logging.

    Sets up logging with appropriate format and level.

    Args:
        log_level: Log level (default from settings)

    Example:
        >>> setup_logging()
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    if log_level is None:
        log_level = settings.log_level

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Suppress verbose logging from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("cohere").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Global settings instance
settings = get_settings()
