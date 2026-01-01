"""Question model for user inquiries about book content."""
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class Question(BaseModel):
    """Represents a user's inquiry about book content.

    Attributes:
        question: The user's question text (1-500 characters)
        selected_text: Optional text highlighted/selected by user (0-5000 characters)
        message_id: Unique identifier for this message (UUID format)
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The user's question about book content"
    )

    selected_text: Optional[str] = Field(
        None,
        max_length=5000,
        description="Text highlighted/selected by user for context-aware query"
    )

    message_id: str = Field(
        ...,
        description="Unique identifier for this message (UUID format)"
    )

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, v: str) -> str:
        """Validate that message_id looks like a UUID."""
        if len(v) < 32:
            raise ValueError("message_id must be a valid UUID string")
        return v

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "question": "How does RAG retrieval work?",
                "selected_text": "RAG (Retrieval-Augmented Generation) combines...",
                "message_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
