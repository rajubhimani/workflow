from functools import lru_cache

from app.config import settings

from .base import VectorStoreBackend


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreBackend:
    backend = settings.VECTOR_STORE_BACKEND.lower()
    if backend == "chroma":
        from .chroma_backend import ChromaVectorStoreBackend

        return ChromaVectorStoreBackend()
    raise ValueError(f"Unsupported vector store backend: {backend}")
