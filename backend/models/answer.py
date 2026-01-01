"""Answer model for chatbot responses to user questions."""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Citation(BaseModel):
    """Citation to a specific book section used in the answer.

    Attributes:
        chapter: Chapter reference
        section: Section within chapter (optional)
        chunk_id: Unique chunk identifier
        relevance_score: Semantic similarity score (0.0-1.0)
    """

    chapter: str = Field(
        ...,
        description="Chapter reference"
    )

    section: Optional[str] = Field(
        None,
        description="Section within chapter (if applicable)"
    )

    chunk_id: int = Field(
        ...,
        description="Unique chunk identifier"
    )

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score from retrieval"
    )

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                    "chapter": "Chapter 1",
                    "section": "RAG Architecture",
                    "chunk_id": 42,
                    "relevance_score": 0.89
                }
        }


class Answer(BaseModel):
    """Represents the system's response to a question.

    Attributes:
        content: The generated answer text (1-2000 characters)
        citations: References to book sections used
        confidence: Confidence score for answer quality (0.0-1.0)
        is_from_book: Whether answer derived from book content
        related_message_id: ID of question this answers
        timestamp: When answer was generated
    """

    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Complete answer text"
    )

    citations: List[Citation] = Field(
        ...,
        description="References to book sections used"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for answer quality"
    )

    is_from_book: bool = Field(
        ...,
        description="Whether answer was derived from book content"
    )

    related_message_id: str = Field(
        ...,
        description="ID of question this answer responds to"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When answer was generated"
    )

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
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
                    "related_message_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2025-12-30T14:30:15Z"
                }
        }

    @classmethod
    def from_book_content(
        cls,
        content: str,
        citations: List[Citation],
        confidence: float,
        related_message_id: str
    ) -> "Answer":
        """Create an Answer derived from book content."""
        return cls(
            content=content,
            citations=citations,
            confidence=confidence,
            is_from_book=True,
            related_message_id=related_message_id
        )

    @classmethod
    def not_from_book(
        cls,
        content: str,
        related_message_id: str
    ) -> "Answer":
        """Create an Answer not derived from book content."""
        return cls(
            content=content,
            citations=[],
            confidence=0.0,
            is_from_book=False,
            related_message_id=related_message_id
        )
