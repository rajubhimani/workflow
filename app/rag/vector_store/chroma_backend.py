import chromadb
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

from app.config import settings
from app.rag.embeddings import get_embeddings

from .base import VectorStoreBackend


class ChromaVectorStoreBackend(VectorStoreBackend):
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    def get_store(self, collection_name: str) -> VectorStore:
        return Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=get_embeddings(),
        )

    def delete_collection(self, collection_name: str) -> None:
        self._client.delete_collection(name=collection_name)

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]
