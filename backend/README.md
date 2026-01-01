# RAG Chatbot Backend

A modular, self-contained FastAPI backend that enables readers to ask questions about book content and receive accurate answers based on RAG (Retrieval-Augmented Generation).

## Features

- Real-time chat via WebSocket
- Zero-hallucination answers (book content only)
- Citation-backed responses with specific book section references
- Selected-text context queries
- Multi-turn conversation with context management
- Deterministic outputs (temperature=0.0)

## Architecture

```
backend/
├── main.py                  # FastAPI server
├── agent.py                 # RAG agent (OpenAI Agents)
├── ingest.py                # Content ingestion script
├── config.py                # Configuration management
├── models/                  # Pydantic models
│   ├── question.py
│   ├── answer.py
│   └── conversation.py
├── services/                # Business logic
│   ├── qdrant_service.py
│   ├── embedding_service.py
│   └── retrieval_service.py
├── api/                     # Routes
│   └── routes.py
└── tests/                   # Tests
```

## Prerequisites

- Python 3.11+
- Qdrant Cloud account (Free Tier)
- Cohere API key
- OpenAI API key

## Installation

1. Clone repository:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

See `.env.example` for all available settings:

| Setting | Description | Default |
|---------|-------------|----------|
| `QDRANT_URL` | Qdrant Cloud URL | Required |
| `QDRANT_API_KEY` | Qdrant API key | Required |
| `COHERE_API_KEY` | Cohere API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_TEMPERATURE` | LLM temperature | 0.0 (deterministic) |
| `MAX_SESSION_MESSAGES` | Max messages per session | 10 |
| `SESSION_TIMEOUT_MINUTES` | Session inactivity timeout | 30 |

## Usage

### 1. Ingest Book Content

Index book content into Qdrant:

```bash
python ingest.py --input-path ../book-content --collection-name book_content
```

Options:
- `--input-path`: Path to markdown files
- `--collection-name`: Qdrant collection name
- `--chunk-size`: Tokens per chunk (default: 700)
- `--chunk-overlap`: Token overlap (default: 100)
- `--batch-size`: Embeddings per request (default: 96)

### 2. Start Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server will be available at `ws://localhost:8000/ws/chat`

### 3. Chat via WebSocket

Connect and send questions:

```python
import asyncio
import websockets
import json

async def chat():
    async with websockets.connect("ws://localhost:8000/ws/chat") as ws:
        # Receive session ID
        session = await ws.recv()

        # Send question
        await ws.send(json.dumps({
            "type": "question",
            "question": "How does RAG work?",
            "selected_text": None,
            "message_id": "uuid..."
        }))

        # Receive streamed response
        async for message in ws:
            msg = json.loads(message)
            if msg["type"] == "answer_chunk":
                print(msg["content"], end="")
            elif msg["type"] == "answer_complete":
                break

asyncio.run(chat())
```

## API Endpoints

### WebSocket: `/ws/chat`

Real-time chat connection.

**Message Types**:

#### Client → Server: `question`
```json
{
  "type": "question",
  "question": "How does RAG retrieval work?",
  "selected_text": "Optional selected text...",
  "message_id": "uuid..."
}
```

#### Server → Client: `answer_chunk`
```json
{
  "type": "answer_chunk",
  "content": "RAG retrieval works by",
  "related_message_id": "uuid..."
}
```

#### Server → Client: `answer_complete`
```json
{
  "type": "answer_complete",
  "content": "RAG retrieval works by querying...",
  "citations": [
    {
      "chapter": "Chapter 1",
      "section": "RAG Architecture",
      "chunk_id": 42,
      "relevance_score": 0.89
    }
  ],
  "confidence": 0.87,
  "is_from_book": true,
  "related_message_id": "uuid..."
}
```

### GET `/health`

Check service connectivity.

**Response**:
```json
{
  "status": "healthy",
  "services": {
    "qdrant": "connected",
    "cohere": "connected",
    "openai": "connected"
  },
  "timestamp": "2025-12-30T14:30:00Z"
}
```

## Constitution Compliance

This backend follows project constitution:

- **Zero Hallucination**: Answers use ONLY indexed book content
- **Citation-Backed**: All responses include [Chapter, Section] references
- **Deterministic**: LLM temperature=0.0, fixed sampling
- **Book-Content Only**: No external knowledge sources
- **Self-Contained**: All code in backend/ with requirements.txt

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_integration.py
```

## Troubleshooting

### Qdrant connection failed
- Verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
- Check Qdrant Cloud dashboard that cluster is running

### Cohere API rate limit
- Reduce `--batch-size` in ingestion
- Wait and re-run ingestion

### Answers not citing book content
- Verify ingestion completed successfully
- Check Qdrant collection has points: `curl https://<url>/collections/<name>`

## Development

See [quickstart.md](../specs/002-rag-chatbot/quickstart.md) for detailed setup guide.

## License

See project LICENSE file.
