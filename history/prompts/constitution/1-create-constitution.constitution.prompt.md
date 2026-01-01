---
id: 1
title: create-constitution
stage: constitution
date: 2025-12-17
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution,spec-first,documentation]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```
/sp.constitution

Project: AI/Spec-Driven Technical Book with Embedded RAG Chatbot

Purpose:
Build a unified technical book using Docusaurus, authored via Spec-Kit Plus and Claude Code, deployed on GitHub Pages, with an embedded RAG chatbot that answers questions strictly from the book content or user-selected text.

Core Principles:
- Spec-first, reproducible development
- Factual accuracy and zero hallucination
- Clear structure for a technical audience
- Full alignment between book content and chatbot knowledge

Content Standards:
- Modular, chapter-based Markdown book
- Clear objectives, explanations, and summaries per chapter
- Defined terminology and no speculative claims

Technical Standards:
- Book: Docusaurus + GitHub Pages
- Chatbot: FastAPI, OpenAI Agents/ChatKit, Qdrant Cloud (Free), Neon Serverless Postgres
- Knowledge source limited to book content only
- Deterministic, citation-backed responses
- Must support answers from user-selected text only

Constraints:
- Public, self-contained GitHub repository
```

## Response snapshot

Created the project constitution file with all specified principles and standards. The constitution includes core principles for spec-first development, factual accuracy, clear structure, alignment between book and chatbot, public repository requirements, and deterministic responses. It also defines technical standards, content standards, development workflow, and governance procedures.

## Outcome

- ✅ Impact: Created initial project constitution at .specify/memory/constitution.md with version 1.0.0
- 🧪 Tests: none
- 📁 Files: 1 file created (.specify/memory/constitution.md)
- 🔁 Next prompts: Plan feature implementation based on constitution
- 🧠 Reflection: Constitution properly captures all specified requirements and principles

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A