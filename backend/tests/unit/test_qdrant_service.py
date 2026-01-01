"""Unit tests for Qdrant service (T019)."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
from services.qdrant_service import QdrantService


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    with patch('services.qdrant_service.QdrantClient') as mock:
        client = Mock()
        mock.return_value = client
        yield client


def test_search_returns_relevant_chunks(mock_qdrant_client):
    """Test that search returns relevant chunks (T019).

    Given: Mock Qdrant client with search results
    When: Searching for relevant content
    Then: Service returns scored points with relevance scores
    """
    # Arrange
    mock_search_result = [
        ScoredPoint(
            id=1,
            version=0,
            score=0.95,
            payload={
                "text": "RAG retrieval works by querying a vector database",
                "chapter": "Chapter 1",
                "section": "RAG Architecture",
                "chunk_id": 42
            },
            vector=None
        ),
        ScoredPoint(
            id=2,
            version=0,
            score=0.87,
            payload={
                "text": "Semantic search uses embeddings to find similar content",
                "chapter": "Chapter 1",
                "section": "Vector Search",
                "chunk_id": 43
            },
            vector=None
        )
    ]

    mock_qdrant_client.search.return_value = mock_search_result

    # Create service with mock client
    with patch('services.qdrant_service.settings') as mock_settings:
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_api_key = "test-key"
        mock_settings.qdrant_collection_name = "test_collection"

        service = QdrantService()
        service.client = mock_qdrant_client

    # Act
    query_vector = [0.1] * 1024  # Mock embedding
    results = service.search(
        collection_name="test_collection",
        query_vector=query_vector,
        limit=5
    )

    # Assert
    assert len(results) == 2
    assert results[0].score == 0.95
    assert results[0].payload["chapter"] == "Chapter 1"
    assert results[0].payload["chunk_id"] == 42
    assert results[1].score == 0.87

    # Verify search was called with correct parameters
    mock_qdrant_client.search.assert_called_once()
    call_args = mock_qdrant_client.search.call_args
    assert call_args[1]["collection_name"] == "test_collection"
    assert call_args[1]["limit"] == 5


def test_create_collection_if_not_exists(mock_qdrant_client):
    """Test collection creation when it doesn't exist.

    Given: Qdrant client without collection
    When: Creating or retrieving collection
    Then: Collection is created with correct vector configuration
    """
    # Arrange
    mock_qdrant_client.collection_exists.return_value = False

    with patch('services.qdrant_service.settings') as mock_settings:
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_api_key = "test-key"
        mock_settings.qdrant_collection_name = "test_collection"

        service = QdrantService()
        service.client = mock_qdrant_client

    # Act
    service.create_collection_if_not_exists(
        collection_name="test_collection",
        vector_size=1024
    )

    # Assert
    mock_qdrant_client.collection_exists.assert_called_with("test_collection")
    mock_qdrant_client.create_collection.assert_called_once()

    # Verify correct vector configuration
    call_args = mock_qdrant_client.create_collection.call_args
    assert call_args[1]["collection_name"] == "test_collection"
    assert call_args[1]["vectors_config"].size == 1024
    assert call_args[1]["vectors_config"].distance == Distance.COSINE


def test_check_health_returns_true_when_connected(mock_qdrant_client):
    """Test health check returns True when connected.

    Given: Connected Qdrant client
    When: Checking health
    Then: Returns True
    """
    # Arrange
    mock_qdrant_client.get_collections.return_value = Mock()

    with patch('services.qdrant_service.settings') as mock_settings:
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_api_key = "test-key"
        mock_settings.qdrant_collection_name = "test_collection"

        service = QdrantService()
        service.client = mock_qdrant_client

    # Act
    is_healthy = service.check_health()

    # Assert
    assert is_healthy is True
    mock_qdrant_client.get_collections.assert_called_once()


def test_check_health_returns_false_on_error(mock_qdrant_client):
    """Test health check returns False on connection error.

    Given: Qdrant client with connection error
    When: Checking health
    Then: Returns False
    """
    # Arrange
    mock_qdrant_client.get_collections.side_effect = Exception("Connection failed")

    with patch('services.qdrant_service.settings') as mock_settings:
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_api_key = "test-key"
        mock_settings.qdrant_collection_name = "test_collection"

        service = QdrantService()
        service.client = mock_qdrant_client

    # Act
    is_healthy = service.check_health()

    # Assert
    assert is_healthy is False
