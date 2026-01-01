<!-- SYNC IMPACT REPORT:
Version change: N/A (initial creation) → 1.0.0
Added sections: All sections (initial constitution creation)
Templates requiring updates: N/A (initial creation)
Follow-up TODOs: None
-->
# AI/Spec-Driven Technical Book with Embedded RAG Chatbot Constitution

## Core Principles

### Spec-First, Reproducible Development
Every aspect of the project starts with clear specifications before implementation; All features must be spec-defined, testable, and documented before coding begins; Clear requirements and acceptance criteria required - no implementation without proper specification.

### Factual Accuracy and Zero Hallucination
All content must be factually accurate and verifiable; The RAG chatbot must only respond with information from the book content or user-selected text; No speculative claims or fabricated information allowed in either book content or chatbot responses.

### Clear Structure for Technical Audience
Content must be organized in a modular, chapter-based format suitable for technical readers; Each chapter must include clear objectives, detailed explanations, and comprehensive summaries; Defined terminology and consistent technical language throughout.

### Full Alignment Between Book Content and Chatbot Knowledge
The chatbot's knowledge base must be strictly limited to the book content only; Responses must be deterministic and citation-backed with references to specific book sections; All chatbot responses must align perfectly with the documented book content.

### Public, Self-Contained Repository
The entire project must be hosted in a public GitHub repository with all necessary dependencies and documentation; No external proprietary components that would limit accessibility or reproduction; Complete setup and deployment instructions must be provided.

### Deterministic, Citation-Backed Responses
All chatbot responses must include citations to specific book sections; Responses must be reproducible and consistent for identical queries; No probabilistic or variable responses that could lead to inconsistency.

## Technical Standards

- Book Platform: Docusaurus framework for static site generation
- Deployment: GitHub Pages for hosting
- Chatbot Framework: FastAPI backend with OpenAI Agents/ChatKit
- Vector Database: Qdrant Cloud (Free tier)
- Database: Neon Serverless Postgres for metadata
- Knowledge Source: Limited to book content only (no external sources)
- User Interaction: Support answers from user-selected text only

## Content Standards

- Format: Modular, chapter-based Markdown book
- Structure: Each chapter must include objectives, explanations, and summaries
- Terminology: Defined and consistent technical terms throughout
- Claims: No speculative or unverified statements allowed
- Updates: Content changes must be reflected in chatbot knowledge base

## Development Workflow

- Spec-Driven: All features must be specified in specs/ before implementation
- Testing: Unit tests for all components, integration tests for chatbot functionality
- Review Process: Code reviews must verify compliance with constitution principles
- Quality Gates: Automated checks for hallucination prevention and content accuracy
- Deployment: Automated deployment pipeline from GitHub Actions

## Governance

This constitution governs all aspects of the project development and takes precedence over any conflicting practices; Amendments require formal documentation, team approval, and migration plan if applicable; All pull requests and reviews must verify compliance with these principles; Complexity must be justified with clear benefits; Use this constitution as the primary guidance for all development decisions.

**Version**: 1.0.0 | **Ratified**: 2025-12-17 | **Last Amended**: 2025-12-17