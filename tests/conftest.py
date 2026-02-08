from unittest.mock import MagicMock, patch

import pytest


FAKE_USAGE_METADATA = {
    "input_tokens": 10,
    "output_tokens": 5,
    "total_tokens": 15,
}


def make_fake_response(content, usage_metadata=None):
    resp = MagicMock()
    resp.content = content
    resp.usage_metadata = usage_metadata if usage_metadata is not None else FAKE_USAGE_METADATA
    return resp


@pytest.fixture(autouse=True)
def clear_sessions():
    from app.api import chat

    chat.sessions.clear()
    yield
    chat.sessions.clear()


@pytest.fixture(autouse=True)
def clear_kb_registry():
    from app.api import knowledge_base

    knowledge_base.kb_registry.clear()
    yield
    knowledge_base.kb_registry.clear()


@pytest.fixture
def mock_llm():
    with patch("app.api.chat.llm") as mock:
        yield mock


@pytest.fixture
def mock_vector_store():
    """Mock the get_vector_store factory to avoid real ChromaDB."""
    mock_backend = MagicMock()
    mock_store = MagicMock()
    mock_backend.get_store.return_value = mock_store
    mock_backend.list_collections.return_value = []
    mock_store.similarity_search_with_score.return_value = []

    with patch("app.api.knowledge_base.get_vector_store", return_value=mock_backend) as _:
        with patch("app.rag.retriever.get_vector_store", return_value=mock_backend):
            yield mock_backend, mock_store


@pytest.fixture
def mock_embeddings():
    with patch("app.rag.embeddings.get_embeddings") as mock:
        mock.return_value = MagicMock()
        yield mock
