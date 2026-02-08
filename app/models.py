from datetime import datetime

from pydantic import BaseModel


# --- Chat models ---


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    knowledge_base_ids: list[str] | None = None


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime
    usage: TokenUsage


class Message(BaseModel):
    role: str
    content: str


class StreamChunk(BaseModel):
    session_id: str
    content: str
    done: bool
    timestamp: datetime
    usage: TokenUsage | None = None


class SessionMessages(BaseModel):
    session_id: str
    messages: list[Message]


# --- Knowledge Base models ---


class CreateKnowledgeBaseRequest(BaseModel):
    name: str
    description: str = ""


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str
    document_count: int
    created_at: datetime


class KnowledgeBaseQueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class RetrievedDocument(BaseModel):
    content: str
    metadata: dict
    score: float


class KnowledgeBaseQueryResponse(BaseModel):
    results: list[RetrievedDocument]
    query: str


class DocumentUploadResponse(BaseModel):
    message: str
    documents_processed: int
    errors: list["FileError"]


class FileError(BaseModel):
    filename: str
    error: str
