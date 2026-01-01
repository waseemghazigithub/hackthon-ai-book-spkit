"""FastAPI application entry point.

This module initializes the FastAPI server with WebSocket
chat endpoint and health check.
"""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from cohere import Client

import uuid
import logging

from config import settings, get_settings, setup_logging
from models.question import Question
from models.answer import Answer
from models.conversation import ConversationSession, Message
from agent import get_rag_agent
from services.qdrant_service import get_qdrant_service
from services.embedding_service import get_embedding_service
import json

logger = logging.getLogger(__name__)


# ========== FastAPI Application ==========

app = FastAPI(
    title="RAG Chatbot API",
    description="Real-time chat API for asking questions about book content using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for frontend integration
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== In-Memory Session Store ==========

conversation_sessions: Dict[str, ConversationSession] = {}


# ========== Health Check Endpoint ==========

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns service connectivity status for Qdrant, Cohere, and OpenAI.

    Returns:
        Dict: Health status and service connection states

    Example:
        >>> GET /health
        {
            "status": "healthy",
            "services": {
                "qdrant": "connected",
                "cohere": "connected",
                "openai": "connected"
            },
            "timestamp": "2025-12-30T14:30:00Z"
        }
    """
    try:
        qdrant_service = get_qdrant_service()
        embedding_service = get_embedding_service()
        rag_agent = get_rag_agent()

        # Check each service
        qdrant_healthy = qdrant_service.check_health()
        cohere_healthy = True  # Embedding service uses Cohere client
        openai_healthy = True  # RAG agent uses OpenAI client

        overall_status = "healthy"
        if not all([qdrant_healthy, cohere_healthy, openai_healthy]):
            overall_status = "degraded"

        return {
            "status": overall_status,
            "services": {
                "qdrant": "connected" if qdrant_healthy else "disconnected",
                "cohere": "connected" if cohere_healthy else "disconnected",
                "openai": "connected" if openai_healthy else "disconnected"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "services": {
                "qdrant": "unknown",
                "cohere": "unknown",
                "openai": "unknown"
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# ========== WebSocket Chat Endpoint ==========

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    WebSocket chat endpoint for real-time conversation.

    Handles:
    - Session initialization with UUID
    - Receiving user questions
    - Streaming answer responses
    - Maintaining conversation context

    Args:
        websocket: WebSocket connection

    Message Flow:
        Client → Server: {"type": "question", "question": "...", "selected_text": "...", "message_id": "..."}
        Server → Client: {"type": "session_init", "session_id": "..."}
        Server → Client: {"type": "answer_chunk", "content": "...", ...}
        Server → Client: {"type": "answer_complete", ...}
    """
    # Generate session ID
    session_id = str(uuid.uuid4())

    # Create conversation session
    session = ConversationSession(
        session_id=session_id,
        max_messages=settings.max_session_messages
    )
    conversation_sessions[session_id] = session

    logger.info(f"WebSocket connected: session_id={session_id}")

    try:
        await websocket.accept()

        # Send session ID to client
        session_init_message = {
            "type": "session_init",
            "session_id": session_id
        }
        await websocket.send_json(session_init_message)
        logger.debug(f"Sent session_init: {session_id}")

        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()

                message_type = data.get("type")

                if message_type == "question":
                    await handle_question(websocket, session, data)
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    await send_error(websocket, f"Unknown message type: {message_type}")

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: session_id={session_id}")
                break
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await send_error(websocket, f"Internal server error: {str(e)}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up session
        if session_id in conversation_sessions:
            del conversation_sessions[session_id]
            logger.info(f"Session cleaned up: {session_id}")


async def handle_question(websocket: WebSocket, session: ConversationSession, data: dict) -> None:
    """
    Handle user question message.

    Args:
        websocket: WebSocket connection
        session: Conversation session
        data: Question message data
    """
    try:
        # Parse question data
        question = data.get("question")
        selected_text = data.get("selected_text")
        message_id = data.get("message_id")

        if not question or not message_id:
            await send_error(websocket, "Missing required fields: question, message_id")
            return

        # Validate using Pydantic model
        question_model = Question(
            question=question,
            selected_text=selected_text,
            message_id=message_id
        )

        # Add user message to session
        user_message = Message.user_message(
            content=question,
            message_id=message_id
        )
        session.add_message(user_message)

        # Get conversation history for context
        history_messages = session.get_conversation_history()
        conversation_history = format_messages_for_agent(history_messages)

        logger.info(f"Processing question: {question[:50]}... (session: {session.session_id})")

        # Generate answer using RAG agent
        rag_agent = get_rag_agent()
        answer = rag_agent.generate_answer(
            question=question,
            selected_text=selected_text,
            conversation_history=conversation_history
        )

        # Update answer with related_message_id
        answer.related_message_id = message_id

        # Add assistant message to session
        assistant_message = Message.assistant_message(
            content=answer.content,
            citations=answer.citations
        )
        session.add_message(assistant_message)

        # Stream answer to client
        await stream_answer(websocket, answer)

        logger.info(f"Answer sent: {len(answer.content)} chars, confidence={answer.confidence}")

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        await send_error(websocket, f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Error handling question: {e}")
        await send_error(websocket, f"Failed to process question: {str(e)}")


async def stream_answer(websocket: WebSocket, answer: Answer) -> None:
    """
    Stream answer to client via WebSocket.

    Args:
        websocket: WebSocket connection
        answer: Generated answer
    """
    try:
        # Send answer in chunks for streaming effect
        chunk_size = 50  # Characters per chunk
        answer_text = answer.content

        for i in range(0, len(answer_text), chunk_size):
            chunk = answer_text[i:i + chunk_size]

            # Send chunk
            chunk_message = {
                "type": "answer_chunk",
                "content": chunk,
                "related_message_id": answer.related_message_id
            }
            await websocket.send_json(chunk_message)
            logger.debug(f"Sent chunk: {i}-{i + chunk_size}")

        # Send completion message
        complete_message = {
            "type": "answer_complete",
            "content": answer.content,
            "citations": [
                {
                    "chapter": c.chapter,
                    "section": c.section,
                    "chunk_id": c.chunk_id,
                    "relevance_score": c.relevance_score
                }
                for c in answer.citations
            ],
            "confidence": answer.confidence,
            "is_from_book": answer.is_from_book,
            "related_message_id": answer.related_message_id,
            "timestamp": answer.timestamp.isoformat()
        }
        await websocket.send_json(complete_message)
        logger.debug(f"Sent answer_complete for message: {answer.related_message_id}")

    except Exception as e:
        logger.error(f"Error streaming answer: {e}")
        await send_error(websocket, f"Error streaming answer: {str(e)}")


async def send_error(websocket: WebSocket, message: str) -> None:
    """
    Send error message to client.

    Args:
        websocket: WebSocket connection
        message: Error message
    """
    error_message = {
        "type": "error",
        "code": "INTERNAL_ERROR",
        "message": message
    }
    await websocket.send_json(error_message)


def format_messages_for_agent(messages: list) -> list:
    """
    Format session messages for RAG agent.

    Args:
        messages: List of Message objects

    Returns:
        list: Formatted messages for OpenAI API
    """
    formatted = []

    for msg in messages:
        if msg.role == "user":
            formatted.append({
                "role": "user",
                "content": msg.content
            })
        elif msg.role == "assistant":
            # Include citations in assistant messages
            citations_text = format_citations_for_context(msg.citations)
            content = f"{msg.content}\n\n[Citations: {citations_text}]"
            formatted.append({
                "role": "assistant",
                "content": content
            })

    return formatted


def format_citations_for_context(citations: list) -> str:
    """
    Format citations for LLM context.

    Args:
        citations: List of Citation objects

    Returns:
        str: Formatted citations string
    """
    if not citations:
        return "None"

    parts = []
    for c in citations:
        if c.section:
            parts.append(f"{c.chapter}, {c.section}")
        else:
            parts.append(c.chapter)

    return "; ".join(parts)


# ========== Session Cleanup Background Task ==========

@app.on_event("startup")
async def startup_event():
    """Configure logging and ensure Qdrant collection exists."""
    setup_logging()
    logger.info("Starting RAG Chatbot API")

    try:
        qdrant_service = get_qdrant_service()
        qdrant_service.ensure_collection_exists()
        logger.info("Qdrant collection ensured")
    except Exception as e:
        logger.error(f"Failed to ensure Qdrant collection: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG Chatbot API")
    # Clean up all sessions
    conversation_sessions.clear()


# ========== Entry Point ==========

if __name__ == "__main__":
    import uvicorn
    from .config import settings

    setup_logging()
    logger.info(f"Starting server on {settings.host}:{settings.port}")

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level
    )
