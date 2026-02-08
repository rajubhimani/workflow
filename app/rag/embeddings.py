from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.OLLAMA_EMBEDDING_MODEL)
