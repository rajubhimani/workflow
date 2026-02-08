from abc import ABC, abstractmethod

from langchain_core.vectorstores import VectorStore


class VectorStoreBackend(ABC):
    @abstractmethod
    def get_store(self, collection_name: str) -> VectorStore:
        """Return a VectorStore instance for the given collection."""

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection and all its data."""

    @abstractmethod
    def list_collections(self) -> list[str]:
        """Return names of all existing collections."""
