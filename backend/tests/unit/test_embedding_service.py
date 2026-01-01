"""Unit tests for Embedding service (T020)."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from services.embedding_service import EmbeddingService


@pytest.fixture
def mock_cohere_client():
    """Mock Cohere client."""
    with patch('services.embedding_service.Client') as mock:
        client = Mock()
        mock.return_value = client
        yield client


def test_generate_embeddings_returns_correct_dimensions(mock_cohere_client):
    """Test that embedding generation returns 1024-dimensional vectors (T020).

    Given: Mock Cohere client
    When: Generating embeddings for texts
    Then: Returns 1024-dimensional vectors
    """
    # Arrange
    mock_response = Mock()
    mock_response.embeddings = [
        [0.1] * 1024,  # 1024-dimensional vector
        [0.2] * 1024,
        [0.3] * 1024
    ]
    mock_cohere_client.embed.return_value = mock_response

    with patch('services.embedding_service.settings') as mock_settings:
        mock_settings.cohere_api_key = "test-key"
        mock_settings.cohere_model = "embed-multilingual-v3.0"

        service = EmbeddingService()
        service.client = mock_cohere_client

    # Act
    texts = ["How does RAG work?", "What is semantic search?", "Explain embeddings"]
    embeddings = service.generate_embeddings(texts)

    # Assert
    assert len(embeddings) == 3
    assert all(len(emb) == 1024 for emb in embeddings), "All embeddings should be 1024-dimensional"

    # Verify Cohere client was called with correct parameters
    mock_cohere_client.embed.assert_called_once()
    call_args = mock_cohere_client.embed.call_args
    assert call_args[1]["texts"] == texts
    assert call_args[1]["model"] == "embed-multilingual-v3.0"
    assert call_args[1]["input_type"] == "search_query"


def test_generate_embeddings_batch(mock_cohere_client):
    """Test batch embedding generation.

    Given: Large list of texts
    When: Generating embeddings in batches
    Then: Processes in batches and returns all embeddings
    """
    # Arrange
    mock_response = Mock()
    # Simulate batch processing: 150 texts, batch size 96
    # Should make 2 calls: 96 + 54
    batch_1_embeddings = [[0.1] * 1024 for _ in range(96)]
    batch_2_embeddings = [[0.2] * 1024 for _ in range(54)]

    mock_cohere_client.embed.side_effect = [
        Mock(embeddings=batch_1_embeddings),
        Mock(embeddings=batch_2_embeddings)
    ]

    with patch('services.embedding_service.settings') as mock_settings:
        mock_settings.cohere_api_key = "test-key"
        mock_settings.cohere_model = "embed-multilingual-v3.0"
        mock_settings.embedding_batch_size = 96

        service = EmbeddingService()
        service.client = mock_cohere_client

    # Act
    texts = [f"Text {i}" for i in range(150)]
    embeddings = service.generate_embeddings_batch(texts, batch_size=96)

    # Assert
    assert len(embeddings) == 150
    assert all(len(emb) == 1024 for emb in embeddings)
    assert mock_cohere_client.embed.call_count == 2  # Two batches


def test_generate_single_embedding(mock_cohere_client):
    """Test single text embedding generation.

    Given: Single text string
    When: Generating embedding
    Then: Returns single 1024-dimensional vector
    """
    # Arrange
    mock_response = Mock()
    mock_response.embeddings = [[0.5] * 1024]
    mock_cohere_client.embed.return_value = mock_response

    with patch('services.embedding_service.settings') as mock_settings:
        mock_settings.cohere_api_key = "test-key"
        mock_settings.cohere_model = "embed-multilingual-v3.0"

        service = EmbeddingService()
        service.client = mock_cohere_client

    # Act
    text = "What is RAG retrieval?"
    embedding = service.generate_single_embedding(text)

    # Assert
    assert len(embedding) == 1024
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)


def test_generate_embeddings_handles_empty_input(mock_cohere_client):
    """Test embedding generation handles empty input gracefully.

    Given: Empty text list
    When: Generating embeddings
    Then: Returns empty list without calling API
    """
    # Arrange
    with patch('services.embedding_service.settings') as mock_settings:
        mock_settings.cohere_api_key = "test-key"
        mock_settings.cohere_model = "embed-multilingual-v3.0"

        service = EmbeddingService()
        service.client = mock_cohere_client

    # Act
    embeddings = service.generate_embeddings([])

    # Assert
    assert embeddings == []
    mock_cohere_client.embed.assert_not_called()


def test_generate_embeddings_handles_api_error(mock_cohere_client):
    """Test embedding generation handles API errors.

    Given: Cohere API error
    When: Generating embeddings
    Then: Raises appropriate exception
    """
    # Arrange
    mock_cohere_client.embed.side_effect = Exception("API rate limit exceeded")

    with patch('services.embedding_service.settings') as mock_settings:
        mock_settings.cohere_api_key = "test-key"
        mock_settings.cohere_model = "embed-multilingual-v3.0"

        service = EmbeddingService()
        service.client = mock_cohere_client

    # Act & Assert
    with pytest.raises(Exception, match="API rate limit exceeded"):
        service.generate_embeddings(["Test text"])
