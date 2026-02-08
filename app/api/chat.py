import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.globals import set_debug
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, UsageMetadata
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_ollama import ChatOllama

from app.config import settings
from app.models import ChatRequest, ChatResponse, Message, SessionMessages, StreamChunk, TokenUsage
from app.rag.retriever import format_context, retrieve_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

if settings.DEBUG:
    logging.basicConfig(level=logging.DEBUG)
    set_debug(True)

llm = ChatOllama(model=settings.OLLAMA_MODEL)

sessions: dict[str, list[BaseMessage]] = {}


def get_trimmed_messages(history: list[BaseMessage]) -> list[BaseMessage]:
    return trim_messages(
        history,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=settings.MAX_TOKENS,
        start_on="human",
    )


def _build_rag_prefix(
    kb_ids: list[str] | None,
    query: str,
) -> list[BaseMessage]:
    """Retrieve context from one or more knowledge bases and return a SystemMessage."""
    if not kb_ids:
        return []
    all_docs: list[tuple[str, dict, float]] = []
    for kb_id in kb_ids:
        all_docs.extend(retrieve_context(kb_id, query))
    context = format_context(all_docs)
    if not context:
        return []
    return [
        SystemMessage(
            content=(
                "Use the following context to answer the user's question. "
                "If the context doesn't contain relevant information, say so.\n\n"
                f"{context}"
            )
        )
    ]


@router.post("")
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    history = sessions.setdefault(session_id, [])
    history.append(HumanMessage(content=request.message))

    rag_prefix = _build_rag_prefix(request.knowledge_base_ids, request.message)
    trimmed = get_trimmed_messages(history)
    messages = rag_prefix + trimmed

    logger.debug("Sending %d message(s) to LLM (trimmed from %d): %s", len(messages), len(history), messages)
    result = await llm.ainvoke(messages)
    logger.debug("LLM response: %s", result.content)
    history.append(AIMessage(content=str(result.content)))

    meta: UsageMetadata | dict = result.usage_metadata or {}
    usage = TokenUsage(
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
        total_tokens=meta.get("total_tokens", 0),
    )

    return ChatResponse(
        response=str(result.content),
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        usage=usage,
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    history = sessions.setdefault(session_id, [])
    history.append(HumanMessage(content=request.message))

    async def generate():
        collected = []
        total_usage = {}
        rag_prefix = _build_rag_prefix(request.knowledge_base_ids, request.message)
        trimmed = get_trimmed_messages(history)
        messages = rag_prefix + trimmed
        logger.debug("Streaming %d message(s) to LLM (trimmed from %d): %s", len(messages), len(history), messages)
        async for chunk in llm.astream(messages):
            collected.append(str(chunk.content))
            if chunk.usage_metadata:
                total_usage = chunk.usage_metadata
            event = StreamChunk(
                session_id=session_id,
                content=str(chunk.content),
                done=False,
                timestamp=datetime.now(timezone.utc),
            )
            yield f"data: {event.model_dump_json()}\n\n"
        history.append(AIMessage(content="".join(collected)))
        usage = TokenUsage(
            input_tokens=total_usage.get("input_tokens", 0),
            output_tokens=total_usage.get("output_tokens", 0),
            total_tokens=total_usage.get("total_tokens", 0),
        )
        event = StreamChunk(
            session_id=session_id,
            content="",
            done=True,
            timestamp=datetime.now(timezone.utc),
            usage=usage,
        )
        yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str) -> SessionMessages:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [
        Message(
            role="user" if isinstance(msg, HumanMessage) else "assistant",
            content=str(msg.content),
        )
        for msg in sessions[session_id]
    ]
    return SessionMessages(session_id=session_id, messages=messages)
