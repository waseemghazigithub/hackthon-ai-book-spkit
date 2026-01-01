"""Conversation session model for managing chat context."""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from models.answer import Citation


class Message(BaseModel):
    """Represents a single message in a conversation.

    Attributes:
        role: "user" or "assistant"
        content: Message text
        timestamp: When message was created
        message_id: UUID for user messages (optional)
        citations: Citations for assistant messages (optional)
    """

    role: str = Field(
        ...,
        description="Message role (user or assistant)"
    )

    content: str = Field(
        ...,
        description="Message text"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When message was created"
    )

    message_id: Optional[str] = Field(
        None,
        description="UUID for user messages"
    )

    citations: Optional[List[Citation]] = Field(
        None,
        description="Citations for assistant messages"
    )

    @classmethod
    def user_message(cls, content: str, message_id: str) -> "Message":
        """Create a user message."""
        return cls(
            role="user",
            content=content,
            message_id=message_id
        )

    @classmethod
    def assistant_message(
        cls,
        content: str,
        citations: List[Citation]
    ) -> "Message":
        """Create an assistant message."""
        return cls(
            role="assistant",
            content=content,
            citations=citations
        )


class ConversationSession(BaseModel):
    """Represents a continuous interaction between a user and the chatbot.

    Maintains context across multiple questions and answers.
    Implements FIFO pruning to limit session size.

    Attributes:
        session_id: Unique session identifier (UUID)
        messages: Conversation history (max max_messages)
        created_at: Session creation timestamp
        last_activity: Last message timestamp
        max_messages: Maximum messages before pruning (default: 10)
    """

    session_id: str = Field(
        ...,
        description="Unique session identifier (UUID)"
    )

    messages: List[Message] = Field(
        default_factory=list,
        description="Conversation history"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Session creation timestamp"
    )

    last_activity: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last message timestamp"
    )

    max_messages: int = Field(
        default=10,
        description="Maximum messages before pruning (FIFO)"
    )

    def add_message(self, message: Message) -> None:
        """Add a message to the session and update last_activity."""
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
        self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        """Remove oldest messages if session exceeds max_messages.

        Implements FIFO (first-in-first-out) pruning:
        - Removes user+assistant pairs (2 messages at a time)
        - Preserves most recent exchanges
        """
        while len(self.messages) > self.max_messages:
            # Remove oldest user+assistant pair (2 messages)
            # If odd number, remove just the oldest one
            if len(self.messages) >= 2 and self.messages[0].role == "user":
                # Remove pair
                self.messages.pop(0)
                if self.messages[0].role == "assistant":
                    self.messages.pop(0)
            else:
                # Remove single oldest message
                self.messages.pop(0)

    def get_conversation_history(self, max_exchanges: Optional[int] = None) -> List[Message]:
        """Get conversation history for LLM context.

        Args:
            max_exchanges: Maximum number of exchanges to return (optional)

        Returns:
            List[Message]: Chronologically ordered messages
        """
        if max_exchanges:
            # Return last 2*max_exchanges messages (user+assistant pairs)
            return self.messages[-(2 * max_exchanges):]
        return self.messages

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired due to inactivity.

        Args:
            timeout_minutes: Inactivity timeout in minutes

        Returns:
            bool: True if session expired, False otherwise
        """
        expiry_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        return self.last_activity < expiry_time

    def get_message_count(self) -> int:
        """Get the number of messages in the session."""
        return len(self.messages)

    def get_last_user_question(self) -> Optional[Message]:
        """Get the last user message (question)."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message
        return None

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "session_id": "660e9500-f30c-52e5-b827-55776551111",
                "messages": [
                    {
                        "role": "user",
                        "content": "How does RAG retrieval work?",
                        "timestamp": "2025-12-30T14:30:00Z",
                        "message_id": "550e8400-e29b-41d4-a716-446655440000"
                    },
                    {
                        "role": "assistant",
                        "content": "RAG retrieval works by querying...",
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
        }
