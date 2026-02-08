# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python application for building AI/ML workflow agents using LangChain, LangGraph, and FastAPI with local LLM support via Ollama.

## Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py

# Run FastAPI dev server
uv run fastapi dev

# Add a dependency
uv add <package>
```

## Architecture

- **main.py** — Slim entry point (imports from `app.main`)
- **app/** — Main application package
  - **main.py** — FastAPI app instance and router wiring
  - **config.py** — Settings via pydantic-settings
  - **models.py** — All Pydantic request/response schemas
  - **api/** — API layer (FastAPI endpoints)
    - **chat.py** — Chat endpoints with optional RAG context injection
    - **knowledge_base.py** — Knowledge base CRUD, document upload, and query endpoints
  - **rag/** — Retrieval-Augmented Generation layer
    - **embeddings.py** — OllamaEmbeddings singleton
    - **ingest.py** — Document loading and chunking
    - **retriever.py** — Vector store retrieval and context formatting
    - **vector_store/** — Pluggable vector store abstraction
      - **base.py** — VectorStoreBackend ABC
      - **chroma_backend.py** — ChromaDB implementation
      - **factory.py** — Backend factory
  - **agent/** — Agent scaffolds (future LangGraph agent)

## Tech Stack

- **Python 3.13** with **uv** as the package manager
- **FastAPI** — REST API framework (includes Uvicorn via `fastapi[standard]`)
- **LangChain + LangGraph** — LLM chain orchestration and stateful multi-agent workflows
- **LangChain Ollama** — Local LLM inference via Ollama
- **LangChain Chroma** — Vector store for RAG knowledge bases
- **PyPDF** — PDF document loading
