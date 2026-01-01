"""Unit tests for Conversation session management (T042-T043)."""
import pytest
from datetime import datetime, timedelta
from models.conversation import ConversationSession, Message


def test_session_pruning_removes_oldest_messages():
    """Test session pruning removes oldest messages when limit reached (T042).

    Given: Session with max 10 messages
    When: Adding 11th message
    Then: Oldest message is removed (FIFO)
    """
    # Arrange
    session = ConversationSession(session_id="test-session", max_messages=10)

    # Add 10 messages (5 user + 5 assistant pairs)
    for i in range(5):
        user_msg = Message.user_message(content=f"Question {i}", message_id=f"msg-{i}")
        assistant_msg = Message.assistant_message(content=f"Answer {i}", citations=[])
        session.add_message(user_msg)
        session.add_message(assistant_msg)

    # Assert initial state
    assert len(session.messages) == 10
    assert session.messages[0].content == "Question 0"
    assert session.messages[1].content == "Answer 0"

    # Act - Add 11th message (should trigger pruning)
    new_user_msg = Message.user_message(content="Question 5", message_id="msg-5")
    session.add_message(new_user_msg)

    # Assert - Oldest message should be removed
    assert len(session.messages) <= 10  # Should not exceed max
    # First message should now be from Question 1 (Question 0 pair was removed)
    assert "Question 0" not in [msg.content for msg in session.messages]


def test_session_stores_max_10_messages():
    """Test session respects max message limit.

    Given: Session with max 10 messages
    When: Adding many messages
    Then: Never exceeds 10 messages
    """
    # Arrange
    session = ConversationSession(session_id="test-session", max_messages=10)

    # Act - Add 20 messages
    for i in range(20):
        msg = Message.user_message(content=f"Message {i}", message_id=f"msg-{i}")
        session.add_message(msg)

    # Assert
    assert len(session.messages) <= 10
    # Should contain most recent messages
    assert "Message 19" in [msg.content for msg in session.messages]
    # Should not contain very old messages
    assert "Message 0" not in [msg.content for msg in session.messages]


def test_session_timeout_after_inactivity():
    """Test session can detect timeout after inactivity (T043).

    Given: Session with 30-minute timeout
    When: Checking if session is expired
    Then: Returns True if last activity was >30 minutes ago
    """
    # Arrange
    session = ConversationSession(session_id="test-session", max_messages=10)

    # Simulate old last_activity timestamp
    old_timestamp = datetime.utcnow() - timedelta(minutes=31)
    session.last_activity = old_timestamp

    # Act
    is_expired = session.is_expired(timeout_minutes=30)

    # Assert
    assert is_expired is True


def test_session_not_expired_when_recently_active():
    """Test session is not expired when recently active.

    Given: Session with recent activity
    When: Checking if session is expired
    Then: Returns False
    """
    # Arrange
    session = ConversationSession(session_id="test-session", max_messages=10)

    # Recent activity (just now)
    session.last_activity = datetime.utcnow()

    # Act
    is_expired = session.is_expired(timeout_minutes=30)

    # Assert
    assert is_expired is False


def test_session_updates_last_activity_on_message():
    """Test session updates last_activity when message is added.

    Given: Session with initial last_activity
    When: Adding new message
    Then: last_activity is updated to current time
    """
    # Arrange
    session = ConversationSession(session_id="test-session", max_messages=10)
    initial_time = session.last_activity

    # Wait a tiny bit to ensure time difference
    import time
    time.sleep(0.01)

    # Act
    msg = Message.user_message(content="New question", message_id="msg-new")
    session.add_message(msg)

    # Assert
    assert session.last_activity > initial_time


def test_session_maintains_conversation_history():
    """Test session maintains complete conversation history.

    Given: Session with user and assistant messages
    When: Getting conversation history
    Then: Returns all messages in chronological order
    """
    # Arrange
    session = ConversationSession(session_id="test-session", max_messages=10)

    # Add conversation
    msg1 = Message.user_message(content="First question", message_id="msg-1")
    msg2 = Message.assistant_message(content="First answer", citations=[])
    msg3 = Message.user_message(content="Second question", message_id="msg-2")
    msg4 = Message.assistant_message(content="Second answer", citations=[])

    session.add_message(msg1)
    session.add_message(msg2)
    session.add_message(msg3)
    session.add_message(msg4)

    # Act
    history = session.get_conversation_history()

    # Assert
    assert len(history) == 4
    assert history[0].content == "First question"
    assert history[0].role == "user"
    assert history[1].content == "First answer"
    assert history[1].role == "assistant"
    assert history[2].content == "Second question"
    assert history[3].content == "Second answer"


def test_message_creation_with_user_role():
    """Test Message creation for user role.

    Given: User message data
    When: Creating message
    Then: Message has correct role and content
    """
    # Act
    msg = Message.user_message(content="Test question", message_id="test-id")

    # Assert
    assert msg.role == "user"
    assert msg.content == "Test question"
    assert msg.message_id == "test-id"
    assert msg.citations is None
    assert isinstance(msg.timestamp, datetime)


def test_message_creation_with_assistant_role():
    """Test Message creation for assistant role.

    Given: Assistant message data with citations
    When: Creating message
    Then: Message has correct role, content, and citations
    """
    # Arrange
    from models.answer import Citation
    citations = [
        Citation(chapter="Chapter 1", section="Section A", chunk_id=10, relevance_score=0.9)
    ]

    # Act
    msg = Message.assistant_message(content="Test answer", citations=citations)

    # Assert
    assert msg.role == "assistant"
    assert msg.content == "Test answer"
    assert len(msg.citations) == 1
    assert msg.citations[0].chapter == "Chapter 1"
    assert msg.message_id is None  # Assistant messages don't have message_id
    assert isinstance(msg.timestamp, datetime)
