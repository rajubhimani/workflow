# Workflow

An LLM-powered chat API built with FastAPI and LangChain, using local models via Ollama. Includes a knowledge base / RAG system for document-grounded conversations.

## Features

- **Chat endpoint** with session-based conversation history
- **Streaming support** via Server-Sent Events (SSE)
- **Session management** — retrieve past messages by session ID
- **Token trimming** — keeps conversation history within configurable token limits
- **Knowledge Base / RAG** — upload documents (PDF, TXT, Markdown, CSV), chunk and embed them into ChromaDB, and use retrieval-augmented generation in chat
- **Multi-KB chat** — query multiple knowledge bases in a single chat request
- **Pluggable vector store** — ChromaDB by default, swappable to Qdrant/Pinecone via the backend abstraction
- **Configurable** via environment variables or `.env` file

## Tech Stack

- **Python 3.13** with **uv** as the package manager
- **FastAPI** — REST API framework
- **LangChain + LangGraph** — LLM orchestration
- **LangChain Ollama** — local LLM inference via Ollama
- **LangChain Chroma** — vector store for RAG knowledge bases
- **PyPDF** — PDF document loading

## Prerequisites

- [Python 3.13+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally with models pulled:
  - A chat model (default: `llama3.2`)
  - An embedding model (default: `nomic-embed-text`) — required for knowledge base features

## Getting Started

```bash
# Install dependencies
uv sync

# Start the dev server
uv run fastapi dev

# Or run directly
uv run python main.py
```

The API will be available at `http://localhost:8000`. Visit the root URL for links to the interactive docs.

## API Endpoints

### Chat

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Landing page with links to docs |
| `POST` | `/chat` | Send a message and get a response |
| `POST` | `/chat/stream` | Send a message and stream the response (SSE) |
| `GET` | `/chat/{session_id}/messages` | Retrieve conversation history for a session |

Chat requests accept an optional `knowledge_base_ids` field to ground responses in uploaded documents:

```json
{
  "message": "What does the report say about Q4 revenue?",
  "knowledge_base_ids": ["kb-id-1", "kb-id-2"]
}
```

### Knowledge Base

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/knowledge-base` | Create a knowledge base |
| `GET` | `/knowledge-base` | List all knowledge bases |
| `GET` | `/knowledge-base/{kb_id}` | Get knowledge base details |
| `DELETE` | `/knowledge-base/{kb_id}` | Delete a knowledge base and its data |
| `POST` | `/knowledge-base/{kb_id}/documents` | Upload documents (multipart form) |
| `POST` | `/knowledge-base/{kb_id}/query` | Standalone similarity search |

**Supported file types:** `.pdf`, `.txt`, `.md`, `.csv`

## Configuration

Set these via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2` | Ollama chat model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `MAX_TOKENS` | `1024` | Max tokens for conversation history trimming |
| `VECTOR_STORE_BACKEND` | `chroma` | Vector store backend (`chroma`) |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB on-disk storage path |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RAG_TOP_K` | `4` | Number of chunks retrieved per query |
| `UPLOAD_DIR` | `./uploads` | Temporary directory for uploaded files |
| `DEBUG` | `false` | Enable debug logging |

## Project Structure

```
workflow/
├── main.py                        # Slim entry point
├── app/
│   ├── main.py                    # FastAPI app and router wiring
│   ├── config.py                  # Settings via pydantic-settings
│   ├── models.py                  # All Pydantic request/response schemas
│   ├── api/
│   │   ├── chat.py                # Chat endpoints with RAG injection
│   │   └── knowledge_base.py      # KB CRUD, upload, and query endpoints
│   ├── rag/
│   │   ├── embeddings.py          # OllamaEmbeddings singleton
│   │   ├── ingest.py              # Document loading and chunking
│   │   ├── retriever.py           # Vector store retrieval
│   │   └── vector_store/
│   │       ├── base.py            # VectorStoreBackend ABC
│   │       ├── chroma_backend.py  # ChromaDB implementation
│   │       └── factory.py         # Backend factory
│   └── agent/                     # Agent scaffolds (future)
└── tests/
    ├── conftest.py                # Shared test fixtures
    ├── test_chat.py               # Chat endpoint tests
    └── test_knowledge_base.py     # KB endpoint tests
```

## Running Tests

```bash
uv run pytest
```
