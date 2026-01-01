# Research: Integrated RAG Chatbot for Book

**Feature**: 002-rag-chatbot
**Date**: 2025-12-30
**Purpose**: Resolves technical unknowns and documents technology choices for implementation

## Overview

This research document establishes the technical foundation for implementing a RAG-based chatbot that answers questions about book content. Key areas covered:
- RAG architecture patterns for zero-hallucination responses
- Cohere embedding best practices
- Qdrant Cloud integration for vector storage
- OpenAI Agents/ChatKit configuration for deterministic outputs
- FastAPI patterns for real-time chat APIs

---

## Decision 1: RAG Architecture Pattern

**Decision**: Use a hybrid retrieval approach with:
1. Dense vector search (semantic similarity) using Cohere embeddings
2. Metadata filtering for selected-text queries
3. Context window management with max 5-10 retrieved chunks
4. Strict prompt engineering requiring citations for all claims

**Rationale**:
- **Zero Hallucination Requirement (Constitution)**: Dense vector search provides semantic relevance while metadata filtering ensures selected-text queries are properly scoped
- **Performance Goal (SC-002: 30s response)**: Retrieving 5-10 chunks keeps token count manageable for generation while providing sufficient context
- **Citation-Backed (Constitution)**: Requiring citations in the prompt enforces determinism and enables source verification
- **Single Knowledge Source (Constitution)**: No hybrid search with keyword/lexical search reduces complexity and keeps system focused on book content

**Alternatives Considered**:
1. Hybrid (Dense + Sparse/Keyword search): Rejected because adds complexity and may retrieve less relevant external content, increasing hallucination risk
2. Re-ranking with Cross-Encoder: Rejected because adds latency and complexity not justified for single-book scope
3. Full book context in prompt: Rejected because exceeds typical LLM context windows and violates performance goals

**Best Practices Applied**:
- Chunk size: 500-1000 tokens with 100-token overlap (balances context preservation and retrieval precision)
- Embedding model: Cohere embed-multilingual-v3.0 (state-of-the-art for English, handles technical terminology)
- Distance metric: Cosine similarity (standard for semantic search)
- Top-K retrieval: Dynamic based on question complexity (default 5, up to 10 for complex queries)

---

## Decision 2: Cohere Embedding Integration

**Decision**: Use Cohere's Python SDK (`cohere` package) with the `embed-multilingual-v3.0` model for generating text embeddings.

**Rationale**:
- **Constitution Alignment**: Cohere provides high-quality embeddings that capture semantic meaning accurately, reducing retrieval errors
- **Technical Terms Handling**: The embed-multilingual-v3.0 model excels at technical terminology, important for technical book content
- **API Reliability**: Cohere's API has high uptime and consistent response times, supporting SC-002 (30s response goal)
- **Batch Processing**: Cohere supports batch embedding generation (up to 96 texts per request), speeding up initial indexing

**Alternatives Considered**:
1. OpenAI text-embedding-3: Rejected because Cohere was specifically mentioned in constraints; also more expensive for large-scale indexing
2. Sentence Transformers (Hugging Face): Rejected because requires local model hosting, adding deployment complexity and violating "self-contained" requirement

**Best Practices Applied**:
- Embedding dimension: 1024 (default for embed-multilingual-v3.0, balances accuracy and storage)
- Input truncation: 512 tokens (covers most chunks while maintaining consistency)
- Batch size: 96 texts (Cohere's maximum per request)
- Error handling: Retry with exponential backoff (handles rate limits)
- Caching: Cache embeddings for unchanged content during development

---

## Decision 3: Qdrant Cloud Configuration

**Decision**: Use Qdrant Cloud Free Tier with the following configuration:
- Collection per book (supports single-book scope)
- HNSW index (Hierarchical Navigable Small World - fast approximate search)
- Payload indexing for metadata (chapter, section, page)
- Replication factor: 1 (Free Tier limitation)

**Rationale**:
- **Free Tier Constraint**: Qdrant Cloud Free Tier provides 1GB storage, sufficient for single technical book (estimated ~500k vectors × 4KB = 2MB actual data)
- **Performance Goal (SC-005: 10 concurrent users)**: HNSW index provides sub-100ms search latency even under concurrent load
- **Selected Text Queries (FR-003/FR-004)**: Payload filtering enables efficient retrieval from specific passages
- **Constitution (Self-Contained)**: Cloud service eliminates local database setup, maintaining self-contained backend

**Alternatives Considered**:
1. Pinecone Free Tier: Rejected because Qdrant was specifically mentioned in constraints and Qdrant offers better payload filtering
2. Local Qdrant instance: Rejected because adds Docker/infrastructure complexity; cloud Free Tier sufficient for scope

**Best Practices Applied**:
- Collection naming: `{book_id}` (supports potential multi-book expansion)
- Vector size: 1024 (matches Cohere embedding dimension)
- Distance: Cosine (standard for semantic embeddings)
- Payload schema:
  ```json
  {
    "chapter": "string",
    "section": "string",
    "page": "integer",
    "chunk_id": "integer"
  }
  ```
- Indexing: HNSW with m=16, ef_construct=100 (balanced for recall vs. speed)

---

## Decision 4: OpenAI Agents/ChatKit Configuration

**Decision**: Use OpenAI's Agents SDK to orchestrate RAG pipeline with deterministic generation settings:
- Model: GPT-4o-mini (cost-effective, fast, sufficient for RAG)
- Temperature: 0.0 (deterministic outputs, constitution requirement)
- Max tokens: 500 (sufficient for concise answers with citations)
- System prompt: Explicitly constrain responses to retrieved context
- Tool use: Use function calling for retrieval (if supported) or pre-retrieval approach

**Rationale**:
- **Constitution (Deterministic, Citation-Backed)**: Temperature 0.0 ensures consistent responses to identical queries
- **Performance Goal (SC-002: 30s response)**: GPT-4o-mini is ~10x faster than GPT-4 while maintaining quality for RAG use case
- **Zero Hallucination (Constitution)**: System prompt explicitly requires citations and prohibits external knowledge
- **Conversational Context (FR-006)**: Agents SDK handles conversation history management natively

**Alternatives Considered**:
1. Direct OpenAI API Chat Completion: Rejected because Agents SDK provides better tool orchestration for retrieval-augmented workflows
2. LangChain: Rejected because adds abstraction layer and complexity; OpenAI Agents is more direct for this use case

**Best Practices Applied**:
- System prompt template:
  ```
  You are a helpful assistant answering questions about a technical book.
  Answer ONLY using the provided book excerpts. Do not use external knowledge.
  Include citations in format [Chapter X, Section Y] for all factual claims.
  If no relevant information is found in the excerpts, state this clearly.
  Keep answers concise and focused on the user's question.
  ```
- Message history: Maintain last 10 exchanges (balances context and performance)
- Token limit: 4000 context window (leaves room for retrieved chunks)

---

## Decision 5: FastAPI Chat API Pattern

**Decision**: Implement WebSocket-based chat API for real-time interaction with the following design:
- Endpoint: `/ws/chat` (WebSocket connection)
- Message format: JSON with `type`, `content`, `selected_text` (optional), `message_id`
- Response streaming: Stream token-by-token for perceived responsiveness
- Session management: In-memory session store (no persistence per spec out-of-scope)
- CORS: Enable for frontend integration

**Rationale**:
- **Real-Time Experience (User Story 3)**: WebSocket enables streaming responses, providing immediate feedback
- **Performance (SC-002)**: Streaming improves perceived latency even if total response time remains ~30s
- **Selected Text (FR-003/FR-004)**: Message format supports optional selected text field for context-aware queries
- **Concurrent Users (SC-005)**: FastAPI's async support handles WebSocket connections efficiently

**Alternatives Considered**:
1. REST API with polling: Rejected because poor UX for chat; streaming is expected
2. Server-Sent Events (SSE): Rejected because WebSocket supports bidirectional messaging (future-proofing)
3. GraphQL subscriptions: Rejected because adds complexity not needed for single-endpoint use case

**Best Practices Applied**:
- Connection lifecycle:
  - Connect → Initialize session → Exchange messages → Disconnect (cleanup)
- Message types:
  - `question`: User question with optional selected text
  - `answer_chunk`: Streaming response token(s)
  - `answer_complete`: Final response with citations
  - `error`: Error message
- Rate limiting: 10 requests/minute per session (prevents abuse)
- Session timeout: 30 minutes of inactivity (cleanup memory)

---

## Decision 6: Conversation Context Management

**Decision**: Implement session-based context management with the following approach:
- Session identifier: UUID generated on connection
- Context window: Last 10 Q&A exchanges
- Context storage: In-memory dict with session ID as key
- Context pruning: FIFO (first-in-first-out) when limit reached
- Topic detection: Simple keyword matching to detect topic shifts

**Rationale**:
- **FR-006 (Conversation Context)**: Maintains context across exchanges without persistence
- **Performance (SC-002)**: In-memory storage is fastest; 10 exchanges balances context and token count
- **SC-004 (80% context retention)**: FIFO ensures recent exchanges are preserved
- **User Story 3 (Topic Shift)**: Keyword detection helps reset context appropriately

**Alternatives Considered**:
1. Redis for session storage: Rejected because adds infrastructure dependency; in-memory sufficient for 10 concurrent users
2. Full conversation history (no pruning): Rejected because would exceed token limits over time
3. Vector-based context similarity: Rejected because adds computational overhead not justified for scope

**Best Practices Applied**:
- Session data structure:
  ```python
  {
    "session_id": "uuid",
    "messages": [
      {"role": "user", "content": "...", "timestamp": "..."},
      {"role": "assistant", "content": "...", "citations": [...]}
    ],
    "created_at": "timestamp"
  }
  ```
- Cleanup: Remove sessions older than 30 minutes (background task)
- Memory limit: ~100KB per session (10 exchanges × 5KB average)

---

## Decision 7: Book Content Ingestion Workflow

**Decision**: Implement `ingest.py` as a standalone script with the following workflow:
1. Parse book content from Markdown files (from Docusaurus structure)
2. Split content into chunks (500-1000 tokens with overlap)
3. Extract metadata (chapter, section headers, page numbers if available)
4. Generate embeddings via Cohere API (batched)
5. Upsert to Qdrant collection
6. Log statistics (chunks indexed, embedding count, time taken)

**Rationale**:
- **Constitution (Book-Content Only)**: Ingestion from source ensures complete, accurate indexing
- **Modularity (FR-012)**: Standalone script allows independent testing and re-running on content updates
- **Performance**: Batch embedding generation reduces API calls and time
- **Metadata Support**: Enables selected-text queries and citation generation

**Alternatives Considered**:
1. Real-time ingestion on startup: Rejected because adds startup latency; content is static
2. External indexing service: Rejected because adds dependency; violates self-contained requirement

**Best Practices Applied**:
- Chunking strategy:
  - Split on paragraph boundaries
  - Target: 700 tokens (±200)
  - Overlap: 100 tokens (preserves context across chunk boundaries)
- Metadata extraction:
  - Chapter: From file names (e.g., `chapter-1.md` → "Chapter 1")
  - Section: From markdown headers (`##`, `###`)
  - Page: None (e-book format doesn't have pages; can use section-level granularity)
- Error handling: Skip malformed chunks, log warnings
- Idempotency: Use `chunk_id` as point ID; re-running updates instead of duplicates

---

## Decision 8: Testing Strategy

**Decision**: Implement comprehensive test coverage using pytest with:
- Unit tests for individual services (Qdrant, embedding, retrieval)
- Integration tests for end-to-end chat flows
- Ingestion tests with sample book content
- Contract tests for API endpoints
- Performance benchmarks for response time

**Rationale**:
- **Spec-Driven Development (Constitution)**: Tests verify requirements before implementation
- **FR-011 (Graceful Error Handling)**: Unit tests cover edge cases (no results, API failures)
- **SC-001 (95% accuracy)**: Integration tests with golden questions validate answer quality
- **Performance Goals**: Benchmarks ensure 30s response time target

**Alternatives Considered**:
1. Manual testing only: Rejected because insufficient for quality assurance and regression prevention
2. End-to-end browser tests: Rejected because backend-focused; frontend is out of scope

**Best Practices Applied**:
- Test structure:
  ```
  tests/
  ├── unit/
  │   ├── test_qdrant_service.py
  │   ├── test_embedding_service.py
  │   └── test_retrieval_service.py
  ├── integration/
  │   ├── test_chat_flow.py
  │   └── test_selected_text.py
  ├── test_ingest.py
  └── conftest.py (fixtures for mock Qdrant, mock Cohere)
  ```
- Mock external services (Qdrant, Cohere, OpenAI) for unit tests
- Integration tests use real services with test collection
- Coverage target: 80%+ (standard for backend services)

---

## Summary of Key Decisions

| Area | Decision | Constitution Alignment |
|------|----------|------------------------|
| RAG Architecture | Dense vector search + metadata filtering | Zero hallucination, citation-backed |
| Embeddings | Cohere embed-multilingual-v3.0 | Accurate semantic understanding |
| Vector DB | Qdrant Cloud Free Tier with HNSW | Self-contained, performance |
| LLM | GPT-4o-mini, temp=0.0 | Deterministic, cost-effective |
| API Pattern | WebSocket streaming | Real-time experience |
| Context | In-memory, 10 exchanges | Session-based (no persistence) |
| Ingestion | Standalone script, batched | Modular, re-runnable |
| Testing | pytest with unit/integration | Spec-driven, quality assurance |

All decisions align with constitution principles and support the success criteria defined in the specification.
