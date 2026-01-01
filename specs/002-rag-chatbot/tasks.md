---

description: "Task list for RAG Chatbot feature implementation"
---

# Tasks: Integrated RAG Chatbot for Book

**Input**: Design documents from `/specs/002-rag-chatbot/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: This feature includes comprehensive test tasks as specified in the success criteria.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend project**: `backend/` at repository root with `src/` implied in backend structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create backend/ directory structure per implementation plan (backend/models/, backend/services/, backend/api/, backend/tests/)
- [X] T002 Create requirements.txt with all dependencies (fastapi, uvicorn, websockets, pydantic, qdrant-client, cohere, openai, python-dotenv, pytest, pytest-asyncio)
- [X] T003 Create .env.example with configuration template (QDRANT_URL, QDRANT_API_KEY, COHERE_API_KEY, OPENAI_API_KEY, etc.)
- [X] T004 [P] Create config.py with Settings class using pydantic-settings for environment variable management
- [X] T005 [P] Create backend/__init__.py and subdirectory __init__.py files (models/, services/, api/, tests/)
- [X] T006 [P] Create README.md with setup instructions and project overview

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T007 Create backend/models/question.py with Question Pydantic model (question, selected_text, message_id fields with validation)
- [X] T008 [P] Create backend/models/answer.py with Answer and Citation Pydantic models (content, citations, confidence, is_from_book, etc.)
- [X] T009 [P] Create backend/models/conversation.py with ConversationSession and Message Pydantic models
- [X] T010 Implement backend/services/qdrant_service.py with QdrantClient initialization, collection creation/retrieval methods
- [X] T011 [P] Implement backend/services/embedding_service.py with Cohere client and batch embedding generation
- [X] T012 Implement backend/services/retrieval_service.py with semantic search using Qdrant and optional metadata filtering
- [X] T013 Create backend/tests/conftest.py with shared fixtures (mock Qdrant, mock Cohere, mock OpenAI)
- [X] T014 Configure logging infrastructure in config.py with appropriate log levels and formatters

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Ask General Questions About Book Content (Priority: P1) 🎯 MVP

**Goal**: Enable readers to submit text-based questions and receive accurate answers based on book material

**Independent Test**: Ask variety of questions about book topics and verify answers are accurate and derived from book content. Delivers immediate value by reducing time spent searching for information.

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T015 [P] [US1] Create backend/tests/test_integration.py with test_ask_question_returns_answer_with_citations() (Given book content indexed, When user asks question, Then answer contains citations)
- [X] T016 [P] [US1] Create backend/tests/test_integration.py with test_ask_question_not_in_book_returns_is_from_book_false() (Given question not covered, When asking, Then is_from_book=False)
- [X] T017 [P] [US1] Create backend/tests/test_integration.py with test_ask_ambiguous_question_requests_clarification() (Given ambiguous question, When asking, Then system requests clarification)
- [X] T018 [P] [US1] Create backend/tests/test_integration.py with test_ask_about_specific_term_returns_definition_with_citations() (Given term question, When asking, Then answer includes definition with citations)
- [X] T019 [P] [US1] Create backend/tests/unit/test_qdrant_service.py with test_search_returns_relevant_chunks() (mock Qdrant, verify search behavior)
- [X] T020 [P] [US1] Create backend/tests/unit/test_embedding_service.py with test_generate_embeddings_returns_correct_dimensions() (mock Cohere, verify 1024-dim vectors)
- [X] T021 [P] [US1] Create backend/tests/unit/test_retrieval_service.py with test_retrieve_with_no_selected_text_uses_semantic_search() (verify retrieval without filtering)

### Implementation for User Story 1

- [X] T022 [US1] Implement backend/agent.py with RAGAgent class using OpenAI Agents/ChatKit, configured for:
  - Temperature 0.0 (deterministic outputs)
  - System prompt requiring citations and book-content-only responses
  - Context window of 4000 tokens
- [X] T023 [US1] Enhance retrieval_service.py with retrieve_for_question() method that:
  - Takes question text and optional selected_text
  - Performs semantic search via Qdrant (top-K=5 default, up to 10)
  - Returns chunks with relevance scores
- [X] T024 [US1] Implement generate_answer() method in RAGAgent that:
  - Takes retrieved chunks and question
  - Formats retrieved content into LLM context
  - Calls OpenAI API with system prompt
  - Parses response into Answer model with citations
  - Returns confidence score based on retrieval relevance
  - Sets is_from_book=True when citations present, False otherwise
- [X] T025 [US1] Create backend/api/routes.py with WebSocket endpoint /ws/chat:
  - Accepts WebSocket connections
  - Generates session_id (UUID) on connect
  - Receives QuestionMessage (question, selected_text, message_id)
  - Calls agent.generate_answer()
  - Streams response as answer_chunk messages
  - Sends answer_complete with citations when done
  - Sends error messages on failures
- [X] T026 [US1] Create backend/main.py with FastAPI application setup:
  - Initialize app with CORS middleware
  - Mount WebSocket routes
  - Add /health endpoint checking Qdrant/Cohere/OpenAI connectivity
  - Configure uvicorn startup
- [X] T027 [US1] Add error handling for:
  - Qdrant connection failures
  - Cohere API rate limits (retry with exponential backoff)
  - OpenAI API errors
  - Malformed user input (validation via Pydantic)
- [X] T028 [US1] Add logging for:
  - WebSocket connection lifecycle
  - Question-answer exchange timing
  - Retrieval statistics (chunks retrieved, relevance scores)
  - Agent generation calls
  - Errors and warnings

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Ask Questions Based on Selected Text (Priority: P1) 🎯 MVP

**Goal**: Enable readers to highlight/select specific text and get contextual answers focused on exact content

**Independent Test**: Select text passages and ask questions, verifying answers are focused on and derived from selected content. Delivers immediate value by enabling context-specific inquiry.

### Tests for User Story 2

- [X] T029 [P] [US2] Create backend/tests/test_integration.py with test_question_with_selected_text_answers_primarily_from_selection() (Given selected paragraph, When asking question, Then answer primarily references selected text)
- [X] T030 [P] [US2] Create backend/tests/test_integration.py with test_explain_code_snippet_with_selected_text() (Given code selection, When asking explanation, Then answer explains code with reference)
- [X] T031 [P] [US2] Create backend/tests/test_integration.py with test_explain_diagram_with_selected_text() (Given diagram description, When asking clarification, Then answer interprets visual element)
- [X] T032 [P] [US2] Create backend/tests/test_integration.py with test_selected_text_with_additional_context_augments_from_book() (Given selection needing more context, When asking, Then answer augments with relevant book content)
- [X] T033 [P] [US2] Create backend/tests/unit/test_retrieval_service.py with test_retrieve_with_selected_text_filters_by_metadata() (verify metadata filtering works)

### Implementation for User Story 2

- [X] T034 [US2] Enhance retrieval_service.py with metadata filtering for selected_text:
  - When selected_text provided, extract chapter/section if embedded in selection
  - Add payload filter to Qdrant search query
  - Boost relevance scores for chunks matching selection metadata
  - Fall back to general semantic search if filtering returns insufficient results
- [X] T035 [US2] Update RAGAgent.generate_answer() to handle selected_text context:
  - When selected_text present, add explicit context in system prompt
  - Instruct LLM to prioritize selected_text in responses
  - Include selected_text in citations when applicable
- [X] T036 [US2] Enhance Question model validation:
  - Add validation for selected_text length (max 5000 characters)
  - Handle very short selections (few words) by expanding search radius
  - Handle very long selections (multiple pages) by extracting key passages
- [X] T037 [US2] Update WebSocket routes.py to:
  - Validate selected_text presence in QuestionMessage
  - Pass selected_text to retrieval service
  - Include selected_text metadata in answer citations when available

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Real-Time Interactive Chat Experience (Priority: P2)

**Goal**: Enable conversational back-and-forth with chatbot for follow-up questions and deeper topic exploration

**Independent Test**: Have multi-turn conversation with follow-up questions and verify context is maintained across turns. Delivers value by enabling deeper exploration of topics.

### Tests for User Story 3

- [X] T038 [P] [US3] Create backend/tests/test_integration.py with test_follow_up_question_maintains_context() (Given previous exchange, When asking follow-up, Then system uses previous context)
- [X] T039 [P] [US3] Create backend/tests/test_integration.py with test_multiple_questions_about_same_topic_reference_same_sections() (Given topic-focused questions, When asking, Then responses consistently reference same sections)
- [X] T040 [P] [US3] Create backend/tests/test_integration.py with test_topic_shift_resets_context_appropriately() (Given new topic, When asking question, Then system shifts focus appropriately)
- [X] T041 [P] [US3] Create backend/tests/test_integration.py with test_conversation_summary() (Given multiple exchanges, When requesting summary, Then system provides concise recap)
- [X] T042 [P] [US3] Create backend/tests/unit/test_conversation.py with test_session_pruning_removes_oldest_messages() (verify FIFO pruning after 10 messages)
- [X] T043 [P] [US3] Create backend/tests/unit/test_conversation.py with test_session_timeout_after_inactivity() (verify session cleanup after 30 min)

### Implementation for User Story 3

- [X] T044 [US3] Implement ConversationSession in-memory store in backend/models/conversation.py:
  - Dict with session_id as key
  - Each session stores list of Message objects (max 10)
  - FIFO pruning when limit reached (remove oldest user+assistant pair)
  - Created_at and last_activity timestamps
- [X] T045 [US3] Create background task for session cleanup:
  - Iterate through sessions every 5 minutes
  - Remove sessions inactive for >30 minutes
  - Log cleanup operations
- [X] T046 [US3] Update WebSocket routes.py with session management:
  - On connect: create new ConversationSession
  - On receive: append user message to session
  - On send: append assistant message to session
  - Pass conversation history to RAGAgent for context
  - Include session context in retrieval (boost relevance for previously discussed topics)
- [X] T047 [US3] Enhance RAGAgent.generate_answer() with conversation context:
  - Accept optional conversation history (last 10 exchanges max)
  - Format history into LLM context (system role: summary, user/assistant roles)
  - Handle topic shift detection (simple keyword matching, optional enhancement)
  - Maintain consistency across responses
- [X] T048 [US3] Add summarization capability:
  - When user requests "summary", generate from conversation history
  - Return concise recap with key topics and citations
  - Limit to 2000 characters (same as regular answers)

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Content Ingestion (Supports All User Stories)

**Goal**: Index book content into Qdrant for retrieval

**Independent Test**: Run ingest.py and verify book content is indexed with correct metadata and embeddings.

### Tests for Ingestion

- [X] T049 [P] [INGEST] Create backend/tests/test_ingest.py with test_parse_markdown_files() (Given book content directory, When parsing, Then extracts chapters and sections correctly)
- [X] T050 [P] [INGEST] Create backend/tests/test_ingest.py with test_chunk_content() (Given markdown content, When chunking, Then produces 500-1000 token chunks with overlap)
- [X] T051 [P] [INGEST] Create backend/tests/test_ingest.py with test_extract_metadata_from_headers() (Given markdown with headers, When extracting, Then identifies chapter/section correctly)
- [X] T052 [P] [INGEST] Create backend/tests/test_ingest.py with test_generate_embeddings() (Given chunks, When generating embeddings, Then produces 1024-dim vectors via Cohere)
- [X] T053 [P] [INGEST] Create backend/tests/test_ingest.py with test_upsert_to_qdrant() (Given embeddings and metadata, When upserting, Then creates points with correct payload)
- [X] T054 [INGEST] Create backend/tests/test_ingest.py with test_ingest_end_to_end() (Given sample book, When running ingest, Then all chunks indexed successfully)

### Implementation for Ingestion

- [X] T055 [INGEST] Create backend/ingest.py with command-line argument parsing (--input-path, --collection-name, --chunk-size, --chunk-overlap, --batch-size)
- [X] T056 [INGEST] Implement parse_book_content() function:
  - Recursively find all .md files in input-path
  - Extract chapter name from filename (e.g., chapter-1.md → "Chapter 1")
  - Parse markdown headers (##, ###) for sections
  - Split content into chunks (500-1000 tokens, configurable)
  - Add 100-token overlap between chunks (configurable)
  - Extract metadata (chapter, section, chunk_id)
- [X] T057 [INGEST] Implement generate_embeddings_batch() function:
  - Use Cohere SDK embed API with batch support
  - Process chunks in batches (default 96 per request)
  - Handle rate limits with exponential backoff retry
  - Return 1024-dimensional vectors
- [X] T058 [INGEST] Implement index_to_qdrant() function:
  - Initialize QdrantClient with credentials
  - Create collection if not exists (HNSW index, cosine distance)
  - Upsert points with vectors and payloads
  - Use chunk_id as point_id for idempotency
  - Log statistics (chunks indexed, time taken)
- [X] T059 [INGEST] Wire up main() execution flow:
  - Parse CLI arguments
  - Parse book content → chunks
  - Generate embeddings for chunks
  - Index to Qdrant
  - Print summary and exit with status code

**Checkpoint**: Book content can be ingested and is ready for retrieval by all user stories

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T060 [P] Update backend/README.md with complete usage instructions matching quickstart.md
- [X] T061 [P] Add docstrings to all public functions and classes
- [X] T062 [P] Add type hints throughout codebase (Python 3.11+)
- [ ] T063 Optimize retrieval performance:
  - Cache frequently accessed book sections
  - Tune top-K parameter based on performance testing
  - Monitor and log retrieval latency
- [X] T064 Security hardening:
  - Ensure no secrets hardcoded (all from .env)
  - Add input sanitization for selected_text
  - Validate all Pydantic models
- [X] T065 [P] Add comprehensive error handling for edge cases:
  - Empty book content (no files to ingest)
  - Very short/very long selected text
  - Ambiguous/poorly-phrased questions
  - Questions about topics not in book
  - Qdrant/Cohere/OpenAI service outages
- [X] T066 [P] Create backend/tests/test_api.py with health endpoint tests (connectivity to all services)
- [ ] T067 Run quickstart.md validation:
  - Follow all 7 steps in quickstart.md
  - Verify health endpoint works
  - Test WebSocket connection and question-answer flow
  - Verify citations are included
  - Test selected-text queries
- [ ] T068 Performance benchmarking:
  - Measure end-to-end response time for SC-002 (30-second goal)
  - Test with 10 concurrent connections for SC-005
  - Monitor memory usage during concurrent sessions
- [X] T069 Add .gitignore for backend/ (venv/, .env, __pycache__/, *.pyc, .pytest_cache/)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2)
- **Ingestion (Phase 6)**: Depends on Foundational (Phase 2) - Supports all user stories but can be done independently
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Extends US1 retrieval but independently testable
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Extends US1/US2 with session management but independently testable
- **Ingestion**: Can start after Foundational (Phase 2) - Independent of user stories

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Models (T007-T009) before services (T010-T012)
- Services before endpoints/routes
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] (T004-T006) can run in parallel
- All Foundational tasks marked [P] (T008-T009, T011) can run in parallel (within Phase 2)
- All test tasks for a user story marked [P] can run in parallel
- User Stories 1, 2, 3 can be worked on in parallel by different team members after Foundational phase
- Ingestion can be developed in parallel with user stories after Foundational phase

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Create backend/tests/test_integration.py with test_ask_question_returns_answer_with_citations()"
Task: "Create backend/tests/test_integration.py with test_ask_question_not_in_book_returns_is_from_book_false()"
Task: "Create backend/tests/test_integration.py with test_ask_ambiguous_question_requests_clarification()"
Task: "Create backend/tests/test_integration.py with test_ask_about_specific_term_returns_definition_with_citations()"
Task: "Create backend/tests/unit/test_qdrant_service.py with test_search_returns_relevant_chunks()"
Task: "Create backend/tests/unit/test_embedding_service.py with test_generate_embeddings_returns_correct_dimensions()"
Task: "Create backend/tests/unit/test_retrieval_service.py with test_retrieve_with_no_selected_text_uses_semantic_search()"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T006)
2. Complete Phase 2: Foundational (T007-T014) **CRITICAL - blocks all stories**
3. Complete Phase 3: User Story 1 (T015-T028)
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Run ingest.py to index book content
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Run ingestion for actual book content
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (T015-T028)
   - Developer B: User Story 2 (T029-T037)
   - Developer C: User Story 3 (T038-T048)
   - Developer D: Ingestion (T049-T059)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (TDD approach)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- Constitution compliance: All answers must include citations (FR-007), use book content only (FR-002), be deterministic (temperature=0.0)
- Performance targets: 30-second response time (SC-002), 10+ concurrent users (SC-005)

---

## Constitution Compliance Checklist

Each task implementation must verify:

- [ ] Zero Hallucination: Answers use ONLY retrieved book content from Qdrant
- [ ] Citation-Backed: All answers include [Chapter X, Section Y] citations
- [ ] Deterministic: LLM configured with temperature=0.0, fixed sampling
- [ ] Book-Content Only: No external knowledge sources in retrieval or generation
- [ ] Self-Contained: All code in backend/ with requirements.txt and .env.example
- [ ] Spec-First: All code follows this tasks.md derived from spec.md and plan.md

---

## Success Criteria Validation

After implementation completion:

- [ ] SC-001: 95% of questions receive answers that accurately reference book content (validate with test suite)
- [ ] SC-002: 90% of question-answer exchanges complete in under 30 seconds (benchmark with T068)
- [ ] SC-003: 85% user satisfaction for comprehension (user acceptance testing)
- [ ] SC-004: 80% of follow-up questions maintain appropriate context (US3 tests)
- [ ] SC-005: System supports 10+ concurrent users without degradation (T068 benchmark)
- [ ] SC-006: 90% of selected-text questions reference selection (US2 tests)
- [ ] SC-007: Session duration decreases 40% vs manual search (user testing)
