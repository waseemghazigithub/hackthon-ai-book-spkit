"""Shared test fixtures for pytest.

This module provides fixtures for mocking external services
(Qdrant, Cohere, OpenAI) and test setup.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Generator, List
from qdrant_client.models import ScoredPoint, PointStruct
import uuid
import os


# ========== Environment Fixtures ==========

@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock environment variables for testing."""
    monkeypatch.setenv("QDRANT_URL", "https://test.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "test-api-key")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "test_book_content")
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.0")


# ========== Qdrant Mock Fixtures ==========

@pytest.fixture
def mock_qdrant_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock QdrantClient for testing."""
    mock_client = Mock()

    # Mock get_collections
    mock_collections = Mock()
    mock_collections.collections = []
    mock_client.get_collections.return_value = mock_collections

    # Mock create_collection
    mock_client.create_collection.return_value = None

    # Mock search
    mock_client.search.return_value = []

    # Mock upsert
    mock_client.upsert.return_value = Mock(status="completed")

    # Mock get_collection
    mock_collection_info = Mock()
    mock_collection_info.status = "green"
    mock_collection_info.points_count = 100
    mock_collection_info.vectors_count = 100
    mock_collection_info.indexed_vectors_count = 100
    mock_client.get_collection.return_value = mock_collection_info

    # Mock delete_collection
    mock_client.delete_collection.return_value = None

    return mock_client


@pytest.fixture
def sample_qdrant_points() -> List[PointStruct]:
    """Sample Qdrant points for testing."""
    return [
        PointStruct(
            id=1,
            vector=[0.1] * 1024,  # Dummy 1024-dim vector
            payload={
                "chapter": "Chapter 1",
                "section": "Introduction",
                "chunk_id": 1,
                "text": "This is the first chunk of the book content."
            }
        ),
        PointStruct(
            id=2,
            vector=[0.2] * 1024,
            payload={
                "chapter": "Chapter 1",
                "section": "Introduction",
                "chunk_id": 2,
                "text": "This is the second chunk explaining RAG architecture."
            }
        ),
        PointStruct(
            id=42,
            vector=[0.3] * 1024,
            payload={
                "chapter": "Chapter 1",
                "section": "RAG Architecture",
                "chunk_id": 42,
                "text": "RAG (Retrieval-Augmented Generation) combines retrieval and generation."
            }
        )
    ]


@pytest.fixture
def sample_scored_points() -> List[ScoredPoint]:
    """Sample scored points from search results."""
    return [
        ScoredPoint(
            id=1,
            version=0,
            score=0.95,
            payload={
                "chapter": "Chapter 1",
                "section": "Introduction",
                "chunk_id": 1,
                "text": "This is the first chunk of the book content."
            }
        ),
        ScoredPoint(
            id=42,
            version=0,
            score=0.89,
            payload={
                "chapter": "Chapter 1",
                "section": "RAG Architecture",
                "chunk_id": 42,
                "text": "RAG (Retrieval-Augmented Generation) combines retrieval and generation."
            }
        )
    ]


# ========== Cohere Mock Fixtures ==========

@pytest.fixture
def mock_cohere_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock Cohere client for testing."""
    mock_client = Mock()

    # Mock embed response
    mock_embed_response = Mock()
    mock_embed_response.embeddings = [[0.1] * 1024, [0.2] * 1024]
    mock_client.embed.return_value = mock_embed_response

    return mock_client


# ========== OpenAI Mock Fixtures ==========

@pytest.fixture
def mock_openai_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock OpenAI client for testing."""
    mock_client = Mock()

    # Mock chat.completions.create
    mock_chat_response = Mock()
    mock_chat_response.choices = [
        Mock(
            message=Mock(
                content="RAG retrieval works by querying a vector database with semantic similarity."
            )
        )
    ]
    mock_chat_response.id = "chatcmpl-test"
    mock_chat_response.created = 1234567890
    mock_chat_response.model = "gpt-4o-mini"

    mock_client.chat.completions.create.return_value = mock_chat_response

    return mock_client


# ========== Model Fixtures ==========

@pytest.fixture
def sample_question_data() -> dict:
    """Sample question data for testing."""
    return {
        "type": "question",
        "question": "How does RAG retrieval work?",
        "selected_text": None,
        "message_id": str(uuid.uuid4())
    }


@pytest.fixture
def sample_question_with_selected_text() -> dict:
    """Sample question data with selected text."""
    return {
        "type": "question",
        "question": "Explain this code snippet",
        "selected_text": "RAG (Retrieval-Augmented Generation) combines retrieval and generation.",
        "message_id": str(uuid.uuid4())
    }


@pytest.fixture
def sample_answer_data() -> dict:
    """Sample answer data for testing."""
    return {
        "type": "answer_complete",
        "content": "RAG retrieval works by querying a vector database...",
        "citations": [
            {
                "chapter": "Chapter 1",
                "section": "RAG Architecture",
                "chunk_id": 42,
                "relevance_score": 0.89
            }
        ],
        "confidence": 0.87,
        "is_from_book": True,
        "related_message_id": str(uuid.uuid4()),
        "timestamp": "2025-12-30T14:30:15Z"
    }


# ========== Session Fixtures ==========

@pytest.fixture
def sample_session_data() -> dict:
    """Sample session data for testing."""
    return {
        "session_id": str(uuid.uuid4()),
        "messages": [
            {
                "role": "user",
                "content": "How does RAG retrieval work?",
                "timestamp": "2025-12-30T14:30:00Z",
                "message_id": str(uuid.uuid4())
            },
            {
                "role": "assistant",
                "content": "RAG retrieval works by querying a vector database...",
                "timestamp": "2025-12-30T14:30:15Z",
                "citations": [
                    {
                        "chapter": "Chapter 1",
                        "section": "RAG Architecture",
                        "chunk_id": 42,
                        "relevance_score": 0.89
                    }
                ]
            }
        ],
        "created_at": "2025-12-30T14:30:00Z",
        "last_activity": "2025-12-30T14:30:15Z"
    }


# ========== Async Fixtures ==========

@pytest.fixture
async def async_mock_qdrant_client() -> Mock:
    """Async mock Qdrant client."""
    mock_client = Mock()
    mock_client.get_collections.return_value = Mock(collections=[])
    mock_client.search.return_value = []
    mock_client.upsert.return_value = Mock(status="completed")
    return mock_client


@pytest.fixture
async def async_mock_cohere_client() -> Mock:
    """Async mock Cohere client."""
    mock_client = Mock()
    mock_embed_response = Mock()
    mock_embed_response.embeddings = [[0.1] * 1024]
    mock_client.embed.return_value = mock_embed_response
    return mock_client


# ========== Health Check Fixture ==========

@pytest.fixture
def mock_service_health() -> dict:
    """Mock service health check response."""
    return {
        "status": "healthy",
        "services": {
            "qdrant": "connected",
            "cohere": "connected",
            "openai": "connected"
        },
        "timestamp": "2025-12-30T14:30:00Z"
    }
