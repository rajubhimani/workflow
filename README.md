# Workflow

An LLM-powered chat API built with FastAPI and LangChain, using local models via Ollama.

## Features

- **Chat endpoint** with session-based conversation history
- **Streaming support** via Server-Sent Events (SSE)
- **Session management** — retrieve past messages by session ID
- **Token trimming** — keeps conversation history within configurable token limits
- **Configurable** via environment variables or `.env` file

## Tech Stack

- **Python 3.13** with **uv** as the package manager
- **FastAPI** — REST API framework
- **LangChain + LangGraph** — LLM orchestration
- **LangChain Ollama** — local LLM inference via Ollama

## Prerequisites

- [Python 3.13+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally with a model pulled (default: `llama3.2`)

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

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Landing page with links to docs |
| `POST` | `/chat` | Send a message and get a response |
| `POST` | `/chat/stream` | Send a message and stream the response (SSE) |
| `GET` | `/chat/{session_id}/messages` | Retrieve conversation history for a session |

## Configuration

Set these via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2` | Ollama model to use |
| `MAX_TOKENS` | `1024` | Max tokens for conversation history trimming |
| `DEBUG` | `false` | Enable debug logging |

## Project Structure

```
workflow/
├── main.py              # Application entry point
├── api/
│   ├── chat.py          # Chat API endpoints
│   ├── chat_models.py   # Pydantic request/response models
│   └── config.py        # Settings via pydantic-settings
└── tests/
    └── test_chat.py     # API tests
```

## Running Tests

```bash
uv run pytest
```
