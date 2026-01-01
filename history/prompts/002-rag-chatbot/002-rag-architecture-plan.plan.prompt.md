---
id: 002
title: RAG Architecture Plan
stage: plan
date: 2025-12-30
surface: agent
model: claude-sonnet-4-5-20250929
feature: 002-rag-chatbot
branch: 002-rag-chatbot
user: AQEEL AHMED
command: /sp.plan
labels: ["rag", "architecture", "implementation", "chatbot", "plan"]
links:
  spec: specs/002-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/002-rag-chatbot/plan.md
 - specs/002-rag-chatbot/research.md
 - specs/002-rag-chatbot/data-model.md
 - specs/002-rag-chatbot/contracts/api.yaml
 - specs/002-rag-chatbot/quickstart.md
 - CLAUDE.md
tests:
 - Constitution check: All gates PASSED
 - Research: 8 technical decisions documented
 - Data model: 4 entities defined with validation rules
 - API contracts: OpenAPI 3.0.3 specification
 - Quickstart: Complete setup guide with troubleshooting
---

## Prompt

Integrated RAG Chatbot for Book

Goal:
- Embed a RAG-based chatbot in the published book to answer reader questions.

Tasks:
1. **Backend Setup**
   - Create `backend/` folder
   - Add `main.py` → FastAPI server
   - Add `agent.py` → OpenRouter/OpenAI agent
   - Add `ingest.py` → Generate embeddings using Cohere API and save to Qdrant

2. **Data Ingestion**
   - Extract book text
   - Generate embeddings via Cohere API
   - Index selected text in Qdrant for retrieval

3. **Chatbot Integration**
   - API endpoint for user questions
   - Return answers using agent + retrieved content
   - Real-time Q&A in book interface

4. **Testing**
   - Validate question-answer accuracy
   - Ensure selected-text-based responses work

Constraints:
- Use Python, FastAPI, OpenAI Agents/ChatKit, Qdrant Free Tier, Cohere API
- Keep backend modular, self-contained

Not included:
- Frontend UI styling
- Multi-user authentication
- Offline database

## Response snapshot

**Created Implementation Plan** for RAG chatbot backend with complete architecture design:

**Phase 0 (Research)**: 8 key technical decisions documented:
- RAG Architecture: Dense vector search + metadata filtering for zero-hallucination
- Embeddings: Cohere embed-multilingual-v3.0 model with batch processing
- Vector DB: Qdrant Cloud Free Tier with HNSW index
- LLM: GPT-4o-mini, temp=0.0 for deterministic outputs
- API: WebSocket streaming for real-time chat
- Context: In-memory session management with 10-message FIFO
- Ingestion: Standalone script with chunked processing
- Testing: pytest with unit/integration tests

**Phase 1 (Design)**: Complete technical artifacts:
- Data model: 4 entities (Question, Answer, BookContent, ConversationSession) with validation rules
- API contracts: OpenAPI 3.0.3 specification for WebSocket and health endpoints
- Quickstart guide: 7-step setup with troubleshooting

**Constitution Check**: All 7 gates PASSED (Spec-First, Zero Hallucination, Clear Structure, Alignment, Self-Contained, Citation-Backed, Deterministic)

**Project Structure**: Modular backend with `main.py`, `agent.py`, `ingest.py`, models/, services/, api/, tests/

**Agent Context**: Updated CLAUDE.md with Python 3.11+, FastAPI, Qdrant, Cohere dependencies

## Outcome

- ✅ Impact: Created comprehensive implementation plan for RAG chatbot backend with zero-hallucination guarantees and citation-backed responses
- 🧪 Tests: Constitution check PASSED (7/7), all technical decisions aligned with spec requirements
- 📁 Files: Created plan.md, research.md, data-model.md, api.yaml, quickstart.md; updated CLAUDE.md
- 🔁 Next prompts: `/sp.tasks` to generate implementation tasks, or begin coding backend components
- 🧠 Reflection: Successfully completed Phase 0 and Phase 1 planning. All architecture decisions documented with rationale and constitutional alignment. Ready for implementation task breakdown.

## Evaluation notes (flywheel)

- Failure modes observed: None - all gates passed
- Graders run and results (PASS/FAIL):
  - Constitution Check: PASS (7/7)
  - Research Completeness: PASS (8 decisions)
  - Data Model Validation: PASS (4 entities)
  - API Contract Definition: PASS (OpenAPI 3.0.3)
  - Quickstart Completeness: PASS (7 steps + troubleshooting)
- Prompt variant (if applicable): Standard /sp.plan workflow
- Next experiment (smallest change to try): Proceed to /sp.tasks for implementation task generation
