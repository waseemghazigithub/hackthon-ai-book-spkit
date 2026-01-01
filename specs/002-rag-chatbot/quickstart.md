# Quickstart Guide: Integrated RAG Chatbot for Book

**Feature**: 002-rag-chatbot
**Date**: 2025-12-30
**Purpose**: Step-by-step guide to set up and run the RAG chatbot backend

## Prerequisites

- Python 3.11 or higher
- Qdrant Cloud account (Free Tier)
- Cohere API key
- OpenAI API key (for GPT-4o-mini)
- Git repository access

---

## Step 1: Clone Repository and Navigate to Backend

```bash
# Clone repository (if not already done)
git clone <repository-url>
cd <repository-root>

# Verify branch
git branch --show-current  # Should be: 002-rag-chatbot
```

---

## Step 2: Install Dependencies

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### `requirements.txt` Contents

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
qdrant-client>=1.7.0
cohere>=4.51.0
openai>=1.3.0
python-dotenv>=1.0.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

---

## Step 3: Configure Environment Variables

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your actual API keys and configuration:

```env
# Qdrant Cloud Configuration
QDRANT_URL=https://<your-qdrant-cluster>.qdrant.io
QDRANT_API_KEY=<your-qdrant-api-key>
QDRANT_COLLECTION_NAME=book_content

# Cohere API (Embeddings)
COHERE_API_KEY=<your-cohere-api-key>
COHERE_MODEL=embed-multilingual-v3.0

# OpenAI API (Chat Generation)
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.0

# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Session Configuration
MAX_SESSION_MESSAGES=10
SESSION_TIMEOUT_MINUTES=30
```

### Getting API Keys

| Service | How to Get |
|---------|-------------|
| Qdrant Cloud | Sign up at [cloud.qdrant.io](https://cloud.qdrant.io), create cluster, get API key from dashboard |
| Cohere | Sign up at [cohere.com](https://cohere.com), get API key from dashboard |
| OpenAI | Sign up at [platform.openai.com](https://platform.openai.com), create API key |

---

## Step 4: Ingest Book Content

Before the chatbot can answer questions, you must index the book content.

```bash
# Run ingestion script
python ingest.py --input-path ../book-content --collection-name book_content
```

### Ingestion Options

| Option | Description | Default |
|--------|-------------|----------|
| `--input-path` | Path to book markdown files | `../book-content` |
| `--collection-name` | Qdrant collection name | `book_content` |
| `--chunk-size` | Tokens per chunk | 700 |
| `--chunk-overlap` | Token overlap between chunks | 100 |
| `--batch-size` | Embeddings per API call | 96 |

### Expected Output

```
Reading book content from: ../book-content
Found 12 chapter files
Chunking content: 2485 chunks (700 tokens average)
Generating embeddings: 26 batches
Indexing to Qdrant: 2485 vectors
✓ Ingestion complete!
  - Chunks indexed: 2485
  - Collection: book_content
  - Time taken: 2m 34s
```

### Book Content Structure

Expected directory structure:
```
book-content/
├── chapter-1.md
├── chapter-2.md
├── chapter-3.md
└── ...
```

Each file should be in Markdown format with headers (`#`, `##`, `###`) for automatic chapter/section extraction.

---

## Step 5: Start the Chatbot Server

```bash
# Start FastAPI server with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Expected Output

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

The server is now running and ready to accept WebSocket connections.

---

## Step 6: Test the API

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
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

### WebSocket Test (Python)

Create a test file `test_chat.py`:

```python
import asyncio
import websockets
import json
import uuid

async def test_chat():
    uri = "ws://localhost:8000/ws/chat"
    message_id = str(uuid.uuid4())

    async with websockets.connect(uri) as websocket:
        # Receive session ID
        session_msg = await websocket.recv()
        print(f"Connected: {session_msg}")

        # Send question
        question = {
            "type": "question",
            "question": "How does RAG retrieval work?",
            "selected_text": None,
            "message_id": message_id
        }
        await websocket.send(json.dumps(question))

        # Receive response
        async for message in websocket:
            msg = json.loads(message)

            if msg["type"] == "answer_chunk":
                print(msg["content"], end="", flush=True)
            elif msg["type"] == "answer_complete":
                print("\n\nCitations:")
                for citation in msg["citations"]:
                    print(f"  - {citation['chapter']}: {citation.get('section', 'N/A')}")
                break

if __name__ == "__main__":
    asyncio.run(test_chat())
```

Run the test:
```bash
python test_chat.py
```

Expected output:
```
Connected: {"type": "session_init", "session_id": "660e9500-..."}
RAG retrieval works by querying a vector database with semantic similarity searches...

Citations:
  - Chapter 1: RAG Architecture
```

---

## Step 7: Integration with Frontend (Optional)

The frontend can connect to the WebSocket endpoint:

```javascript
// Frontend WebSocket connection example
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
  console.log('Connected to chatbot');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'session_init':
      console.log('Session ID:', message.session_id);
      break;

    case 'answer_chunk':
      // Stream content to UI
      appendToChat(message.content);
      break;

    case 'answer_complete':
      // Display citations
      displayCitations(message.citations);
      break;

    case 'error':
      showError(message.message);
      break;
  }
};

// Send question
function askQuestion(question, selectedText = null) {
  ws.send(JSON.stringify({
    type: 'question',
    question: question,
    selected_text: selectedText,
    message_id: crypto.randomUUID()
  }));
}
```

---

## Troubleshooting

### Issue: Qdrant connection failed

**Error**: `Qdrant connection error: Connection refused`

**Solution**:
1. Verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
2. Check Qdrant Cloud dashboard that cluster is running
3. Try `curl https://<qdrant-url>/healthz` to verify connectivity

### Issue: Cohere API rate limit

**Error**: `Cohere API error: Rate limit exceeded`

**Solution**:
1. Reduce `--batch-size` in ingestion (e.g., `--batch-size 48`)
2. Wait and re-run ingestion (Cohere Free Tier has limits)

### Issue: OpenAI API authentication

**Error**: `OpenAI API error: Incorrect API key provided`

**Solution**:
1. Verify `OPENAI_API_KEY` in `.env`
2. Check that API key has access to GPT-4o-mini
3. Ensure no extra spaces or quotes in `.env` file

### Issue: Ingestion is slow

**Cause**: Large book with many chunks

**Solution**:
1. Increase `--batch-size` (up to 96 for Cohere)
2. Run ingestion during off-peak hours
3. Consider splitting book into parts and ingesting separately

### Issue: Answers not citing book content

**Cause**: Content not indexed or retrieval parameters misconfigured

**Solution**:
1. Verify ingestion completed successfully
2. Check Qdrant collection has points: `curl https://<qdrant-url>/collections/book_content`
3. Review logs for retrieval errors

---

## Common Use Cases

### 1. General Question (No Selected Text)

```python
{
  "type": "question",
  "question": "What is RAG?",
  "selected_text": None,
  "message_id": "uuid..."
}
```

### 2. Context-Aware Question (With Selected Text)

```python
{
  "type": "question",
  "question": "Explain this code snippet",
  "selected_text": "```python\ndef retrieve(query):\n    ...",
  "message_id": "uuid..."
}
```

### 3. Follow-up Question (Uses Session Context)

```python
{
  "type": "question",
  "question": "How does that differ from dense retrieval?",
  "selected_text": None,
  "message_id": "uuid..."
}
```

---

## Performance Tips

1. **Reduce Latency**: Increase `--batch-size` during ingestion to reduce API calls
2. **Improve Accuracy**: Adjust retrieval top-K in `config.py` (default 5-10)
3. **Handle Concurrent Users**: FastAPI's async support scales well; monitor with `/health`
4. **Session Cleanup**: Sessions auto-expire after 30 min; adjust in `.env` if needed

---

## Next Steps

- **Run Tests**: `pytest` to verify all components
- **Review Logs**: Check `logs/` directory for server and ingestion logs
- **Integration**: Connect frontend to WebSocket endpoint
- **Monitoring**: Set up alerts for service health (`/health` endpoint)

---

## Support

For issues or questions:
1. Check `backend/README.md` for detailed documentation
2. Review `logs/` directory for error messages
3. Consult `specs/002-rag-chatbot/research.md` for technical details
