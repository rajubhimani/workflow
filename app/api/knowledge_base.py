import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.models import (
    CreateKnowledgeBaseRequest,
    DocumentUploadResponse,
    FileError,
    KnowledgeBaseQueryRequest,
    KnowledgeBaseQueryResponse,
    KnowledgeBaseResponse,
    RetrievedDocument,
)
from app.rag.ingest import SUPPORTED_EXTENSIONS, process_document
from app.rag.retriever import retrieve_context
from app.rag.vector_store import get_vector_store

router = APIRouter(prefix="/knowledge-base", tags=["Knowledge Base"])

# In-memory registry for KB metadata
kb_registry: dict[str, dict] = {}


@router.post("", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(request: CreateKnowledgeBaseRequest):
    kb_id = str(uuid.uuid4())
    kb_registry[kb_id] = {
        "id": kb_id,
        "name": request.name,
        "description": request.description,
        "document_count": 0,
        "created_at": datetime.now(timezone.utc),
    }
    return KnowledgeBaseResponse(**kb_registry[kb_id])


@router.get("", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases():
    return [KnowledgeBaseResponse(**kb) for kb in kb_registry.values()]


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(kb_id: str):
    if kb_id not in kb_registry:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return KnowledgeBaseResponse(**kb_registry[kb_id])


@router.delete("/{kb_id}", status_code=204)
async def delete_knowledge_base(kb_id: str):
    if kb_id not in kb_registry:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    try:
        get_vector_store().delete_collection(kb_id)
    except Exception:
        pass  # Collection may not exist yet if no documents were uploaded
    del kb_registry[kb_id]


@router.post(
    "/{kb_id}/documents",
    response_model=DocumentUploadResponse,
)
async def upload_documents(kb_id: str, files: list[UploadFile]):
    if kb_id not in kb_registry:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    backend = get_vector_store()
    store = backend.get_store(kb_id)

    processed = 0
    errors: list[FileError] = []

    for file in files:
        filename = file.filename or "unnamed"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            errors.append(
                FileError(
                    filename=filename,
                    error=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                )
            )
            continue

        file_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            chunks = await process_document(file_path, filename)
            store.add_documents(chunks)
            processed += len(chunks)
        except Exception as e:
            errors.append(FileError(filename=filename, error=str(e)))
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    kb_registry[kb_id]["document_count"] += processed

    return DocumentUploadResponse(
        message=f"Processed {processed} document chunks",
        documents_processed=processed,
        errors=errors,
    )


@router.post("/{kb_id}/query", response_model=KnowledgeBaseQueryResponse)
async def query_knowledge_base(kb_id: str, request: KnowledgeBaseQueryRequest):
    if kb_id not in kb_registry:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    results = retrieve_context(kb_id, request.query, request.top_k)
    return KnowledgeBaseQueryResponse(
        results=[
            RetrievedDocument(content=content, metadata=metadata, score=score)
            for content, metadata, score in results
        ],
        query=request.query,
    )
