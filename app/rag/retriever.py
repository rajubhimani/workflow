from app.config import settings
from app.rag.vector_store import get_vector_store


def retrieve_context(
    kb_id: str, query: str, top_k: int | None = None
) -> list[tuple[str, dict, float]]:
    """Retrieve relevant documents from a knowledge base.

    Returns list of (content, metadata, score) tuples.
    """
    k = top_k or settings.RAG_TOP_K
    store = get_vector_store().get_store(kb_id)
    results = store.similarity_search_with_score(query, k=k)
    return [(doc.page_content, doc.metadata, score) for doc, score in results]


def format_context(documents: list[tuple[str, dict, float]]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    if not documents:
        return ""
    parts = []
    for i, (content, metadata, _score) in enumerate(documents, 1):
        source = metadata.get("source_filename", "unknown")
        parts.append(f"[Document {i} — {source}]\n{content}")
    return "\n\n".join(parts)
