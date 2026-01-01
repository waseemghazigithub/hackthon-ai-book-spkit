"""Qdrant vector database operations service.

This module handles Qdrant client initialization, collection management,
and vector search operations for retrieving relevant book content.
"""
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
from config import settings, get_settings
import logging

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for Qdrant vector database operations.

    Manages collection creation, upsertion of points,
    and semantic search retrieval.
    """

    def __init__(self, settings_instance: Optional[Any] = None):
        """Initialize Qdrant service.

        Args:
            settings_instance: Optional settings instance (uses global if None)
        """
        self.settings = settings_instance or get_settings()
        self.client: Optional[QdrantClient] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Qdrant client with credentials."""
        try:
            self.client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key
            )
            logger.info(f"Qdrant client initialized: {self.settings.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist.

        Creates collection with HNSW index and cosine distance.
        """
        collection_name = self.settings.qdrant_collection_name

        try:
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]

            if collection_name not in existing_names:
                logger.info(f"Creating collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1024,  # Cohere embed-multilingual-v3.0 dimension
                        distance=Distance.COSINE
                    ),
                )
                logger.info(f"Collection created successfully: {collection_name}")
            else:
                logger.info(f"Collection already exists: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    def upsert_points(
        self,
        points: List[PointStruct]
    ) -> None:
        """Upsert points to Qdrant collection.

        Args:
            points: List of PointStruct with vectors and payloads
        """
        collection_name = self.settings.qdrant_collection_name

        try:
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            logger.info(f"Upserted {len(points)} points to {collection_name}")
            return operation_info
        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            raise

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_: Optional[Filter] = None
    ) -> List[models.ScoredPoint]:
        """Search for similar vectors in Qdrant.

        Args:
            query_vector: Query embedding vector (1024 dimensions)
            limit: Maximum number of results to return (default: 5)
            score_threshold: Minimum similarity score (optional)
            filter_: Payload filter for metadata filtering (optional)

        Returns:
            List[ScoredPoint]: List of scored points with payloads

        Example:
            >>> results = qdrant_service.search(query_vector, limit=5)
            >>> for result in results:
            ...     print(result.score, result.payload['text'])
        """
        collection_name = self.settings.qdrant_collection_name

        try:
            # Use query_points for newer qdrant-client versions (>= 1.7.0)
            search_results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=filter_,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )

            # query_points returns QueryResponse, extract points
            points = search_results.points if hasattr(search_results, 'points') else search_results
            logger.debug(f"Search returned {len(points)} results")
            return points
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information.

        Returns:
            Dict: Collection metadata (count, status, etc.)

        Example:
            >>> info = qdrant_service.get_collection_info()
            >>> print(info['points_count'])
        """
        collection_name = self.settings.qdrant_collection_name

        try:
            info = self.client.get_collection(collection_name)
            return {
                "status": info.status,
                "points_count": info.points_count,
                "vectors_count": getattr(info, 'vectors_count', info.points_count),
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', info.points_count)
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the collection.

        Use with caution - this removes all indexed content.
        """
        collection_name = self.settings.qdrant_collection_name

        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Collection deleted: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def check_health(self) -> bool:
        """Check if Qdrant connection is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Try to get collections list
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False


# Singleton instance
_qdrant_service: Optional[QdrantService] = None


def get_qdrant_service() -> QdrantService:
    """Get singleton Qdrant service instance.

    Returns:
        QdrantService: Singleton instance

    Example:
        >>> service = get_qdrant_service()
        >>> results = service.search(query_vector)
    """
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
