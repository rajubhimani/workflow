import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.documents import Document

from app.main import app


# --- CRUD Tests ---


@pytest.mark.asyncio
async def test_create_knowledge_base():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/knowledge-base",
            json={"name": "Test KB", "description": "A test knowledge base"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test KB"
    assert data["description"] == "A test knowledge base"
    assert data["document_count"] == 0
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_knowledge_base_no_description():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/knowledge-base",
            json={"name": "Minimal KB"},
        )

    assert response.status_code == 201
    assert response.json()["description"] == ""


@pytest.mark.asyncio
async def test_list_knowledge_bases_empty():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/knowledge-base")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_knowledge_bases():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post("/knowledge-base", json={"name": "KB1"})
        await client.post("/knowledge-base", json={"name": "KB2"})

        response = await client.get("/knowledge-base")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    names = {kb["name"] for kb in data}
    assert names == {"KB1", "KB2"}


@pytest.mark.asyncio
async def test_get_knowledge_base():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create_resp = await client.post(
            "/knowledge-base", json={"name": "My KB", "description": "desc"}
        )
        kb_id = create_resp.json()["id"]

        response = await client.get(f"/knowledge-base/{kb_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == kb_id
    assert data["name"] == "My KB"


@pytest.mark.asyncio
async def test_get_knowledge_base_not_found():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/knowledge-base/nonexistent")

    assert response.status_code == 404
    assert response.json()["detail"] == "Knowledge base not found"


@pytest.mark.asyncio
async def test_delete_knowledge_base(mock_vector_store):
    mock_backend, _ = mock_vector_store

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create_resp = await client.post("/knowledge-base", json={"name": "To Delete"})
        kb_id = create_resp.json()["id"]

        response = await client.delete(f"/knowledge-base/{kb_id}")

    assert response.status_code == 204

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        get_resp = await client.get(f"/knowledge-base/{kb_id}")

    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_knowledge_base_not_found(mock_vector_store):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.delete("/knowledge-base/nonexistent")

    assert response.status_code == 404


# --- Upload Tests ---


@pytest.mark.asyncio
async def test_upload_document(mock_vector_store):
    mock_backend, mock_store = mock_vector_store

    with patch("app.api.knowledge_base.process_document", new_callable=AsyncMock) as mock_process:
        mock_process.return_value = [
            Document(page_content="chunk 1", metadata={}),
            Document(page_content="chunk 2", metadata={}),
        ]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            create_resp = await client.post("/knowledge-base", json={"name": "Upload KB"})
            kb_id = create_resp.json()["id"]

            response = await client.post(
                f"/knowledge-base/{kb_id}/documents",
                files=[("files", ("test.txt", b"Hello world", "text/plain"))],
            )

    assert response.status_code == 200
    data = response.json()
    assert data["documents_processed"] == 2
    assert data["errors"] == []
    mock_store.add_documents.assert_called_once()


@pytest.mark.asyncio
async def test_upload_multiple_documents(mock_vector_store):
    mock_backend, mock_store = mock_vector_store

    with patch("app.api.knowledge_base.process_document", new_callable=AsyncMock) as mock_process:
        mock_process.side_effect = [
            [Document(page_content="chunk A", metadata={})],
            [Document(page_content="chunk B", metadata={})],
        ]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            create_resp = await client.post("/knowledge-base", json={"name": "Multi Upload KB"})
            kb_id = create_resp.json()["id"]

            response = await client.post(
                f"/knowledge-base/{kb_id}/documents",
                files=[
                    ("files", ("doc1.txt", b"Content 1", "text/plain")),
                    ("files", ("doc2.md", b"# Content 2", "text/markdown")),
                ],
            )

    assert response.status_code == 200
    data = response.json()
    assert data["documents_processed"] == 2
    assert data["errors"] == []


@pytest.mark.asyncio
async def test_upload_unsupported_file_type(mock_vector_store):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create_resp = await client.post("/knowledge-base", json={"name": "Bad Upload"})
        kb_id = create_resp.json()["id"]

        response = await client.post(
            f"/knowledge-base/{kb_id}/documents",
            files=[("files", ("image.png", b"fake-png", "image/png"))],
        )

    assert response.status_code == 200
    data = response.json()
    assert data["documents_processed"] == 0
    assert len(data["errors"]) == 1
    assert "Unsupported file type" in data["errors"][0]["error"]


@pytest.mark.asyncio
async def test_upload_to_nonexistent_kb(mock_vector_store):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/knowledge-base/nonexistent/documents",
            files=[("files", ("test.txt", b"Hello", "text/plain"))],
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_mixed_valid_and_invalid(mock_vector_store):
    mock_backend, mock_store = mock_vector_store

    with patch("app.api.knowledge_base.process_document", new_callable=AsyncMock) as mock_process:
        mock_process.return_value = [Document(page_content="chunk", metadata={})]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            create_resp = await client.post("/knowledge-base", json={"name": "Mixed"})
            kb_id = create_resp.json()["id"]

            response = await client.post(
                f"/knowledge-base/{kb_id}/documents",
                files=[
                    ("files", ("valid.txt", b"Hello", "text/plain")),
                    ("files", ("bad.exe", b"binary", "application/octet-stream")),
                ],
            )

    assert response.status_code == 200
    data = response.json()
    assert data["documents_processed"] == 1
    assert len(data["errors"]) == 1
    assert data["errors"][0]["filename"] == "bad.exe"


@pytest.mark.asyncio
async def test_upload_updates_document_count(mock_vector_store):
    mock_backend, mock_store = mock_vector_store

    with patch("app.api.knowledge_base.process_document", new_callable=AsyncMock) as mock_process:
        mock_process.return_value = [
            Document(page_content="c1", metadata={}),
            Document(page_content="c2", metadata={}),
            Document(page_content="c3", metadata={}),
        ]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            create_resp = await client.post("/knowledge-base", json={"name": "Count KB"})
            kb_id = create_resp.json()["id"]

            await client.post(
                f"/knowledge-base/{kb_id}/documents",
                files=[("files", ("test.txt", b"Hello", "text/plain"))],
            )

            get_resp = await client.get(f"/knowledge-base/{kb_id}")

    assert get_resp.json()["document_count"] == 3


# --- Query Tests ---


@pytest.mark.asyncio
async def test_query_knowledge_base(mock_vector_store):
    mock_backend, mock_store = mock_vector_store
    mock_store.similarity_search_with_score.return_value = [
        (Document(page_content="Relevant content", metadata={"source_filename": "test.txt"}), 0.85),
    ]

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create_resp = await client.post("/knowledge-base", json={"name": "Query KB"})
        kb_id = create_resp.json()["id"]

        response = await client.post(
            f"/knowledge-base/{kb_id}/query",
            json={"query": "test query"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "Relevant content"
    assert data["results"][0]["score"] == 0.85


@pytest.mark.asyncio
async def test_query_with_custom_top_k(mock_vector_store):
    mock_backend, mock_store = mock_vector_store
    mock_store.similarity_search_with_score.return_value = []

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create_resp = await client.post("/knowledge-base", json={"name": "TopK KB"})
        kb_id = create_resp.json()["id"]

        response = await client.post(
            f"/knowledge-base/{kb_id}/query",
            json={"query": "test", "top_k": 2},
        )

    assert response.status_code == 200
    mock_store.similarity_search_with_score.assert_called_once_with("test", k=2)


@pytest.mark.asyncio
async def test_query_nonexistent_kb(mock_vector_store):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/knowledge-base/nonexistent/query",
            json={"query": "test"},
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_query_empty_results(mock_vector_store):
    mock_backend, mock_store = mock_vector_store
    mock_store.similarity_search_with_score.return_value = []

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create_resp = await client.post("/knowledge-base", json={"name": "Empty KB"})
        kb_id = create_resp.json()["id"]

        response = await client.post(
            f"/knowledge-base/{kb_id}/query",
            json={"query": "nothing here"},
        )

    assert response.status_code == 200
    assert response.json()["results"] == []


# --- Supported file extensions ---


@pytest.mark.asyncio
async def test_upload_pdf(mock_vector_store):
    mock_backend, mock_store = mock_vector_store

    with patch("app.api.knowledge_base.process_document", new_callable=AsyncMock) as mock_process:
        mock_process.return_value = [Document(page_content="pdf content", metadata={})]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            create_resp = await client.post("/knowledge-base", json={"name": "PDF KB"})
            kb_id = create_resp.json()["id"]

            response = await client.post(
                f"/knowledge-base/{kb_id}/documents",
                files=[("files", ("doc.pdf", b"%PDF-fake", "application/pdf"))],
            )

    assert response.status_code == 200
    assert response.json()["documents_processed"] == 1
    assert response.json()["errors"] == []


@pytest.mark.asyncio
async def test_upload_csv(mock_vector_store):
    mock_backend, mock_store = mock_vector_store

    with patch("app.api.knowledge_base.process_document", new_callable=AsyncMock) as mock_process:
        mock_process.return_value = [Document(page_content="csv content", metadata={})]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            create_resp = await client.post("/knowledge-base", json={"name": "CSV KB"})
            kb_id = create_resp.json()["id"]

            response = await client.post(
                f"/knowledge-base/{kb_id}/documents",
                files=[("files", ("data.csv", b"a,b\n1,2", "text/csv"))],
            )

    assert response.status_code == 200
    assert response.json()["documents_processed"] == 1


# --- Delete cleans up vector store ---


@pytest.mark.asyncio
async def test_delete_calls_delete_collection(mock_vector_store):
    mock_backend, _ = mock_vector_store

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create_resp = await client.post("/knowledge-base", json={"name": "Del KB"})
        kb_id = create_resp.json()["id"]

        await client.delete(f"/knowledge-base/{kb_id}")

    mock_backend.delete_collection.assert_called_once_with(kb_id)
