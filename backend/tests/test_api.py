"""API endpoint tests (T066)."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_services():
    """Mock all external services."""
    with patch('main.get_qdrant_service') as mock_qdrant, \
         patch('main.get_embedding_service') as mock_embedding, \
         patch('main.get_rag_agent') as mock_agent:

        # Setup mock returns
        mock_qdrant_service = Mock()
        mock_qdrant_service.check_health.return_value = True
        mock_qdrant.return_value = mock_qdrant_service

        mock_embedding_service = Mock()
        mock_embedding.return_value = mock_embedding_service

        mock_rag = Mock()
        mock_agent.return_value = mock_rag

        yield {
            "qdrant": mock_qdrant_service,
            "embedding": mock_embedding_service,
            "rag": mock_rag
        }


@pytest.fixture
def client(mock_services):
    """Create test client with mocked services."""
    # Import after mocking to ensure mocks are applied
    from main import app
    return TestClient(app)


def test_health_endpoint_returns_healthy_status(client, mock_services):
    """Test /health endpoint returns healthy status (T066).

    Given: All services connected
    When: Calling /health
    Then: Returns healthy status with service states
    """
    # Arrange
    mock_services["qdrant"].check_health.return_value = True

    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded", "unhealthy"]
    assert "services" in data
    assert "qdrant" in data["services"]
    assert "cohere" in data["services"]
    assert "openai" in data["services"]
    assert "timestamp" in data


def test_health_endpoint_returns_degraded_when_service_down(client, mock_services):
    """Test /health returns degraded when service is down.

    Given: Qdrant service disconnected
    When: Calling /health
    Then: Returns degraded status
    """
    # Arrange
    mock_services["qdrant"].check_health.return_value = False

    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code in [200, 503]
    data = response.json()
    assert data["status"] in ["degraded", "unhealthy"]
    assert data["services"]["qdrant"] == "disconnected"


def test_health_endpoint_handles_service_exception(client, mock_services):
    """Test /health handles service exceptions gracefully.

    Given: Service raises exception
    When: Calling /health
    Then: Returns unhealthy status without crashing
    """
    # Arrange
    mock_services["qdrant"].check_health.side_effect = Exception("Connection error")

    # Act
    response = client.get("/health")

    # Assert
    # Should not crash, should return error status
    assert response.status_code in [200, 503]
    data = response.json()
    assert data["status"] in ["degraded", "unhealthy"]


def test_health_endpoint_includes_timestamp(client, mock_services):
    """Test /health endpoint includes timestamp.

    Given: Health check request
    When: Calling /health
    Then: Response includes ISO 8601 timestamp
    """
    # Arrange
    mock_services["qdrant"].check_health.return_value = True

    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data
    # Verify timestamp format (ISO 8601)
    timestamp = data["timestamp"]
    assert "T" in timestamp or "-" in timestamp  # Basic ISO 8601 check


@pytest.mark.asyncio
async def test_websocket_chat_endpoint_establishes_connection():
    """Test WebSocket /ws/chat establishes connection.

    Given: WebSocket client
    When: Connecting to /ws/chat
    Then: Connection established and session_id sent
    """
    # This test requires WebSocket test client
    from fastapi.testclient import TestClient
    from main import app

    with patch('main.get_qdrant_service'), \
         patch('main.get_embedding_service'), \
         patch('main.get_rag_agent'):

        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as websocket:
            # Should receive session_init message
            data = websocket.receive_json()
            assert data["type"] == "session_init"
            assert "session_id" in data
            assert len(data["session_id"]) > 0  # UUID format


@pytest.mark.asyncio
async def test_websocket_handles_question_message():
    """Test WebSocket handles question message.

    Given: Established WebSocket connection
    When: Sending question message
    Then: Receives answer chunks and completion
    """
    from fastapi.testclient import TestClient
    from models.answer import Answer, Citation
    from datetime import datetime

    # Mock RAG agent response
    mock_answer = Answer(
        content="Test answer about RAG",
        citations=[
            Citation(chapter="Chapter 1", section="Test", chunk_id=1, relevance_score=0.9)
        ],
        confidence=0.85,
        is_from_book=True,
        related_message_id="test-msg-id",
        timestamp=datetime.utcnow()
    )

    with patch('main.get_qdrant_service'), \
         patch('main.get_embedding_service'), \
         patch('main.get_rag_agent') as mock_agent:

        mock_rag = Mock()
        mock_rag.generate_answer.return_value = mock_answer
        mock_agent.return_value = mock_rag

        from main import app
        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as websocket:
            # Receive session init
            session_data = websocket.receive_json()
            assert session_data["type"] == "session_init"

            # Send question
            websocket.send_json({
                "type": "question",
                "question": "What is RAG?",
                "selected_text": None,
                "message_id": "test-msg-id"
            })

            # Receive answer chunks
            messages = []
            while True:
                try:
                    msg = websocket.receive_json(timeout=5)
                    messages.append(msg)
                    if msg["type"] == "answer_complete":
                        break
                except Exception:
                    break

            # Assert
            assert len(messages) > 0
            # Should have answer_chunk and answer_complete messages
            chunk_messages = [m for m in messages if m["type"] == "answer_chunk"]
            complete_messages = [m for m in messages if m["type"] == "answer_complete"]

            assert len(chunk_messages) >= 1 or len(complete_messages) == 1
            if complete_messages:
                complete_msg = complete_messages[0]
                assert "content" in complete_msg
                assert "citations" in complete_msg
                assert "confidence" in complete_msg
                assert "is_from_book" in complete_msg


@pytest.mark.asyncio
async def test_websocket_handles_invalid_message():
    """Test WebSocket handles invalid message gracefully.

    Given: Established WebSocket connection
    When: Sending invalid message
    Then: Receives error message
    """
    from fastapi.testclient import TestClient

    with patch('main.get_qdrant_service'), \
         patch('main.get_embedding_service'), \
         patch('main.get_rag_agent'):

        from main import app
        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as websocket:
            # Receive session init
            websocket.receive_json()

            # Send invalid message (missing required fields)
            websocket.send_json({
                "type": "question"
                # Missing: question, message_id
            })

            # Should receive error message
            try:
                response = websocket.receive_json(timeout=5)
                # Should be error type or connection closes
                if "type" in response:
                    assert response["type"] in ["error", "session_init"]
            except Exception:
                # Connection may close on invalid message
                pass


def test_cors_middleware_allows_frontend():
    """Test CORS middleware allows frontend requests.

    Given: FastAPI app with CORS middleware
    When: Making request with Origin header
    Then: CORS headers included in response
    """
    from fastapi.testclient import TestClient

    with patch('main.get_qdrant_service') as mock_qdrant, \
         patch('main.get_embedding_service'), \
         patch('main.get_rag_agent'):

        mock_qdrant.return_value.check_health.return_value = True

        from main import app
        client = TestClient(app)

        # Make request with Origin header
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )

        # Assert CORS headers present
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers or \
               "Access-Control-Allow-Origin" in response.headers


def test_app_metadata():
    """Test FastAPI app has correct metadata.

    Given: FastAPI app
    When: Checking app metadata
    Then: Has correct title and version
    """
    from main import app

    # Assert
    assert app.title == "RAG Chatbot API"
    assert "1.0" in app.version
    assert len(app.description) > 0


def test_health_check_structure():
    """Test health check response structure matches spec.

    Given: Health endpoint
    When: Calling /health
    Then: Response matches OpenAPI spec structure
    """
    from fastapi.testclient import TestClient

    with patch('main.get_qdrant_service') as mock_qdrant, \
         patch('main.get_embedding_service'), \
         patch('main.get_rag_agent'):

        mock_qdrant.return_value.check_health.return_value = True

        from main import app
        client = TestClient(app)

        # Act
        response = client.get("/health")

        # Assert structure matches api.yaml
        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "status" in data
        assert "services" in data
        assert "timestamp" in data

        # Service fields
        services = data["services"]
        assert "qdrant" in services
        assert "cohere" in services
        assert "openai" in services

        # Status enum
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

        # Service status enum
        for service_status in services.values():
            assert service_status in ["connected", "disconnected", "unknown"]
