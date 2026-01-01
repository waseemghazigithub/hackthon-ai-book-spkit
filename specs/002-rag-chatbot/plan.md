# Implementation Plan: Integrated RAG Chatbot for Book

**Branch**: `002-rag-chatbot` | **Date**: 2025-12-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a modular, self-contained backend service that enables readers to ask questions about book content and receive accurate answers based on RAG (Retrieval-Augmented Generation). The system will:
- Index book content using vector embeddings stored in Qdrant
- Provide a FastAPI-based chat interface
- Support both general questions and selected-text-based queries
- Maintain conversation context for multi-turn interactions

Technical approach: FastAPI backend + OpenAI Agents/ChatKit for response generation + Qdrant Cloud for vector storage + Cohere API for embeddings.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: FastAPI, OpenAI Agents/ChatKit, Cohere SDK, Qdrant Client, Uvicorn
**Storage**: Qdrant Cloud (Free Tier) for vector embeddings; file-based for metadata and configuration
**Testing**: pytest with asyncio support
**Target Platform**: Local development (Windows/Linux/Mac) with capability to deploy to cloud
**Project Type**: backend (self-contained API service)
**Performance Goals**: 30-second response time for Q&A exchanges (SC-002); 10+ concurrent users (SC-005)
**Constraints**:
- Modular and self-contained backend service
- Must run locally via uvicorn
- Answers must be based solely on book content (constitution: zero hallucination)
- Must include citations to specific book sections (constitution: citation-backed responses)
- Free-tier services only (Qdrant Free Tier)
- No user authentication (out of scope)
- No persistent conversation history beyond current session (out of scope)
**Scale/Scope**:
- Single book content indexing
- 10+ concurrent users
- Session-based (ephemeral) conversation history
- Modular code structure allowing independent testing

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Spec-First, Reproducible Development
- ✅ **PASS**: This plan is created from spec.md before implementation
- ✅ **PASS**: All requirements and acceptance criteria defined
- ✅ **PASS**: Implementation follows documented plan

### Factual Accuracy and Zero Hallucination
- ✅ **PASS**: System will retrieve ONLY from indexed book content in Qdrant
- ✅ **PASS**: Agent will be configured to limit responses to retrieved context
- ✅ **PASS**: All responses will include citations to book sections

### Clear Structure for Technical Audience
- ✅ **PASS**: Modular backend structure with clear separation (main.py, agent.py, ingest.py)
- ✅ **PASS**: API contracts will be documented with OpenAPI specification
- ✅ **PASS**: Code follows Python naming conventions and typing

### Full Alignment Between Book Content and Chatbot Knowledge
- ✅ **PASS**: Chatbot knowledge base strictly limited to book content
- ✅ **PASS**: No external knowledge sources in retrieval
- ✅ **PASS**: Content updates will require re-indexing via ingest.py

### Public, Self-Contained Repository
- ✅ **PASS**: All code in backend/ directory with requirements.txt
- ✅ **PASS**: Configuration via environment variables (no hardcoded secrets)
- ✅ **PASS**: Setup and deployment instructions included

### Deterministic, Citation-Backed Responses
- ✅ **PASS**: Responses include specific book section references
- ✅ **PASS**: Consistent retrieval using fixed embedding model and search parameters
- ✅ **PASS**: Temperature and sampling configured for deterministic outputs

**All gates PASSED - Proceed to Phase 0**

## Project Structure

### Documentation (this feature)

```text
specs/002-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
│   └── api.yaml         # OpenAPI specification
├── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
└── spec.md              # Feature specification (already exists)
```

### Source Code (repository root)

```text
backend/
├── README.md                    # Setup and usage instructions
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── pyproject.toml               # Project configuration (optional)
├── main.py                      # FastAPI server entry point
├── agent.py                     # RAG agent using OpenAI Agents/ChatKit
├── ingest.py                    # Book content ingestion and indexing
├── config.py                    # Configuration management
├── models/                      # Data models (Pydantic)
│   ├── __init__.py
│   ├── question.py              # Question request model
│   ├── answer.py                # Answer response model
│   └── conversation.py          # Conversation session model
├── services/                    # Business logic
│   ├── __init__.py
│   ├── qdrant_service.py        # Qdrant vector DB operations
│   ├── embedding_service.py     # Cohere embedding generation
│   └── retrieval_service.py     # Semantic search and retrieval
├── api/                         # API routes
│   ├── __init__.py
│   └── routes.py                # Chat endpoint(s)
└── tests/                       # Tests
    ├── __init__.py
    ├── conftest.py              # Shared test fixtures
    ├── test_integration.py      # Integration tests
    ├── test_ingest.py           # Ingestion tests
    └── test_api.py              # API endpoint tests
```

**Structure Decision**: Selected modular backend structure (Option 2 variant). The backend directory will contain all server code, with clear separation:
- `main.py` - FastAPI application setup and server startup
- `agent.py` - RAG agent orchestrating retrieval and generation
- `ingest.py` - Standalone script for indexing book content
- `models/` - Pydantic models for request/response validation
- `services/` - Core business logic for embeddings, retrieval, and Qdrant operations
- `api/` - HTTP endpoint definitions
- `tests/` - Unit and integration tests

This structure supports FR-012 (modular and self-contained) and FR-013 (locally runnable).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

All constitution gates passed. No complexity tracking needed.
