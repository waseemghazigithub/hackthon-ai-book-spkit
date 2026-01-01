"""Cohere embedding generation service.

This module handles text embedding generation using Cohere API
with batch processing and retry logic for rate limits.
"""
from typing import List, Optional
from cohere import Client
from config import settings, get_settings
import logging
import time

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings via Cohere API.

    Supports batch processing, rate limit handling with exponential backoff.
    """

    def __init__(self, settings_instance=None):
        """Initialize Cohere embedding service.

        Args:
            settings_instance: Optional settings instance (uses global if None)
        """
        self.settings = settings_instance or get_settings()
        self.client: Client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Cohere client with API key."""
        try:
            self.client = Client(api_key=self.settings.cohere_api_key)
            logger.info(f"Cohere client initialized with model: {self.settings.cohere_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            raise

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Processes texts in batches to handle rate limits.
        Implements exponential backoff retry.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per batch (default from settings)

        Returns:
            List[List[float]]: List of 1024-dimensional embedding vectors

        Raises:
            Exception: If embedding generation fails after retries

        Example:
            >>> service = EmbeddingService()
            >>> embeddings = service.generate_embeddings(["text1", "text2"])
            >>> print(len(embeddings[0]))  # 1024
        """
        if batch_size is None:
            batch_size = self.settings.embedding_batch_size

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._generate_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        logger.info(f"Generated {len(all_embeddings)} embeddings in {len(texts)} texts")
        return all_embeddings

    def _generate_batch_with_retry(
        self,
        texts: List[str],
        max_retries: int = 3,
        initial_backoff: float = 1.0
    ) -> List[List[float]]:
        """Generate embeddings for a batch with retry logic.

        Implements exponential backoff for rate limits.

        Args:
            texts: Batch of texts to embed
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds

        Returns:
            List[List[float]]: Embedding vectors for the batch

        Raises:
            Exception: If all retries are exhausted
        """
        backoff = initial_backoff

        for attempt in range(max_retries):
            try:
                response = self.client.embed(
                    texts=texts,
                    model=self.settings.cohere_model,
                    input_type="search_document"
                )
                embeddings = response.embeddings

                # Validate embedding dimensions (should be 1024 for embed-multilingual-v3.0)
                if embeddings:
                    expected_dim = len(embeddings[0])
                    if expected_dim != 1024:
                        logger.warning(
                            f"Unexpected embedding dimension: {expected_dim} "
                            f"(expected 1024 for {self.settings.cohere_model})"
                        )

                logger.debug(f"Generated embeddings for batch of {len(texts)} texts")
                return embeddings

            except Exception as e:
                is_last_attempt = attempt == max_retries - 1

                if is_last_attempt:
                    logger.error(f"Failed to generate embeddings after {max_retries} attempts: {e}")
                    raise

                logger.warning(
                    f"Embedding generation attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {backoff}s..."
                )

                time.sleep(backoff)
                backoff *= 2  # Exponential backoff

        # Should never reach here
        raise Exception("Unexpected error in embedding generation retry logic")

    def generate_embedding(
        self,
        text: str
    ) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List[float]: 1024-dimensional embedding vector

        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.generate_embedding("example text")
            >>> print(len(embedding))  # 1024
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton EmbeddingService instance.

    Returns:
        EmbeddingService: Singleton instance

    Example:
        >>> service = get_embedding_service()
        >>> embedding = service.generate_embedding("text")
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
