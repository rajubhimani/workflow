import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from app.main import app
from tests.conftest import FAKE_USAGE_METADATA, make_fake_response


@pytest.mark.asyncio
async def test_chat(mock_llm):
    mock_llm.ainvoke = AsyncMock(return_value=make_fake_response("Hello! How can I help you?"))

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Hello! How can I help you?"
    assert "session_id" in data
    assert "timestamp" in data
    assert data["usage"]["input_tokens"] == 10
    assert data["usage"]["output_tokens"] == 5
    assert data["usage"]["total_tokens"] == 15

    call_args = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(call_args[0], HumanMessage)
    assert call_args[0].content == "Hello"


@pytest.mark.asyncio
async def test_chat_returns_session_id(mock_llm):
    mock_llm.ainvoke = AsyncMock(return_value=make_fake_response("Hi"))

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat", json={"message": "Hello"})

    session_id = response.json()["session_id"]
    assert session_id
    assert len(session_id) == 36  # UUID format


@pytest.mark.asyncio
async def test_chat_multi_turn(mock_llm):
    mock_llm.ainvoke = AsyncMock(side_effect=[
        make_fake_response("Hi there!"),
        make_fake_response("Your name is Bob."),
    ])

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r1 = await client.post("/chat", json={"message": "My name is Bob"})
        session_id = r1.json()["session_id"]

        r2 = await client.post(
            "/chat", json={"message": "What is my name?", "session_id": session_id}
        )

    assert r2.json()["response"] == "Your name is Bob."

    from app.api.chat import sessions

    history = sessions[session_id]
    assert len(history) == 4
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "My name is Bob"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "Hi there!"
    assert isinstance(history[2], HumanMessage)
    assert history[2].content == "What is my name?"
    assert isinstance(history[3], AIMessage)
    assert history[3].content == "Your name is Bob."


@pytest.mark.asyncio
async def test_chat_separate_sessions(mock_llm):
    mock_llm.ainvoke = AsyncMock(side_effect=[
        make_fake_response("Response A"),
        make_fake_response("Response B"),
    ])

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r1 = await client.post("/chat", json={"message": "Session A"})
        r2 = await client.post("/chat", json={"message": "Session B"})

    assert r1.json()["session_id"] != r2.json()["session_id"]

    from app.api.chat import sessions

    s1 = sessions[r1.json()["session_id"]]
    s2 = sessions[r2.json()["session_id"]]
    assert len(s1) == 2
    assert len(s2) == 2
    assert s1[0].content == "Session A"
    assert s2[0].content == "Session B"


@pytest.mark.asyncio
async def test_chat_empty_message(mock_llm):
    mock_llm.ainvoke = AsyncMock(return_value=make_fake_response(""))

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat", json={"message": ""})

    assert response.status_code == 200
    assert response.json()["response"] == ""


@pytest.mark.asyncio
async def test_chat_missing_message():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat", json={})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_stream(mock_llm):
    chunk1 = make_fake_response("Hello", usage_metadata=None)
    chunk2 = make_fake_response(" world", usage_metadata=FAKE_USAGE_METADATA)

    async def fake_astream(messages):
        for chunk in [chunk1, chunk2]:
            yield chunk

    mock_llm.astream = fake_astream

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat/stream", json={"message": "Hi"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.strip().split("\n\n")
    ]
    assert len(events) == 3
    assert events[0]["content"] == "Hello"
    assert events[0]["done"] is False
    assert "timestamp" in events[0]
    assert events[0]["usage"] is None
    assert events[1]["content"] == " world"
    assert events[1]["done"] is False
    assert events[2]["content"] == ""
    assert events[2]["done"] is True
    assert events[2]["usage"]["input_tokens"] == 10
    assert events[2]["usage"]["total_tokens"] == 15
    assert events[0]["session_id"] == events[2]["session_id"]


@pytest.mark.asyncio
async def test_chat_stream_stores_history(mock_llm):
    chunk1 = make_fake_response("Hi")

    async def fake_astream(messages):
        yield chunk1

    mock_llm.astream = fake_astream

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat/stream", json={"message": "Hello"})

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.strip().split("\n\n")
    ]
    session_id = events[0]["session_id"]
    from app.api.chat import sessions

    assert len(sessions[session_id]) == 2
    assert isinstance(sessions[session_id][0], HumanMessage)
    assert isinstance(sessions[session_id][1], AIMessage)
    assert sessions[session_id][1].content == "Hi"


@pytest.mark.asyncio
async def test_chat_stream_missing_message():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat/stream", json={})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_session_messages(mock_llm):
    mock_llm.ainvoke = AsyncMock(return_value=make_fake_response("Hi there!"))

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post("/chat", json={"message": "Hello"})
        session_id = r.json()["session_id"]

        response = await client.get(f"/chat/{session_id}/messages")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert len(data["messages"]) == 2
    assert data["messages"][0] == {"role": "user", "content": "Hello"}
    assert data["messages"][1] == {"role": "assistant", "content": "Hi there!"}


@pytest.mark.asyncio
async def test_get_session_messages_not_found():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/chat/nonexistent/messages")

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


@pytest.mark.asyncio
async def test_home_page():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Workflow API" in response.text
    assert "/docs" in response.text
    assert "/redoc" in response.text


@pytest.mark.asyncio
async def test_chat_with_rag_context(mock_llm, mock_vector_store):
    """Chat with knowledge_base_id should prepend RAG context as a SystemMessage."""
    from langchain_core.documents import Document
    from langchain_core.messages import SystemMessage

    from app.api.knowledge_base import kb_registry

    # Set up a fake KB
    kb_id = "test-kb-id"
    kb_registry[kb_id] = {
        "id": kb_id,
        "name": "Test KB",
        "description": "",
        "document_count": 1,
        "created_at": "2024-01-01T00:00:00Z",
    }

    _mock_backend, mock_store = mock_vector_store
    mock_store.similarity_search_with_score.return_value = [
        (Document(page_content="Paris is the capital of France", metadata={"source_filename": "geo.txt"}), 0.9),
    ]

    mock_llm.ainvoke = AsyncMock(return_value=make_fake_response("Paris"))

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/chat",
            json={"message": "What is the capital of France?", "knowledge_base_ids": [kb_id]},
        )

    assert response.status_code == 200
    assert response.json()["response"] == "Paris"

    # Verify the LLM received a SystemMessage with context
    call_args = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)
    assert "Paris is the capital of France" in call_args[0].content
    assert isinstance(call_args[1], HumanMessage)


@pytest.mark.asyncio
async def test_chat_without_rag_no_system_message(mock_llm):
    """Chat without knowledge_base_id should NOT have a SystemMessage."""
    mock_llm.ainvoke = AsyncMock(return_value=make_fake_response("Hi"))

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    call_args = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(call_args[0], HumanMessage)


@pytest.mark.asyncio
async def test_chat_with_multiple_knowledge_bases(mock_llm, mock_vector_store):
    """Chat with knowledge_base_ids should retrieve context from all KBs."""
    from langchain_core.documents import Document
    from langchain_core.messages import SystemMessage

    from app.api.knowledge_base import kb_registry

    kb1_id = "kb-1"
    kb2_id = "kb-2"
    for kb_id, name in [(kb1_id, "KB1"), (kb2_id, "KB2")]:
        kb_registry[kb_id] = {
            "id": kb_id,
            "name": name,
            "description": "",
            "document_count": 1,
            "created_at": "2024-01-01T00:00:00Z",
        }

    _mock_backend, mock_store = mock_vector_store
    mock_store.similarity_search_with_score.side_effect = [
        [(Document(page_content="France info", metadata={"source_filename": "geo.txt"}), 0.9)],
        [(Document(page_content="History info", metadata={"source_filename": "history.txt"}), 0.8)],
    ]

    mock_llm.ainvoke = AsyncMock(return_value=make_fake_response("Combined answer"))

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/chat",
            json={"message": "Tell me about France", "knowledge_base_ids": [kb1_id, kb2_id]},
        )

    assert response.status_code == 200
    assert response.json()["response"] == "Combined answer"

    call_args = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)
    assert "France info" in call_args[0].content
    assert "History info" in call_args[0].content


