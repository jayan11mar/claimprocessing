"""
Abstract VectorStore interface for the claims knowledge base.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from app.rag.chunkers import Chunk


class VectorStore(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects.
            embeddings: List of embedding vectors corresponding to each chunk.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.

        Args:
            query: The original query string.
            query_embedding: Embedding vector for the query.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of (Chunk, score) tuples.
        """
        pass

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None) -> None:
        """
        Delete entries from the store.

        Args:
            ids: Optional list of source IDs to delete.
                 If None, deletes all entries.
        """
        pass

    @abstractmethod
    def persist(self, path: Optional[str] = None) -> None:
        """
        Persist the store to disk.

        Args:
            path: Optional path to persist to.
        """
        pass

    @abstractmethod
    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Return a retriever interface for this store.

        Args:
            search_kwargs: Optional search configuration.

        Returns:
            A retriever object (LangChain VectorStoreRetriever or equivalent).
        """
        pass


def get_vector_store(backend: str = "faiss", **kwargs) -> VectorStore:
    """
    Get the appropriate vector store based on backend.

    Args:
        backend: Vector store backend ("faiss", "chroma", "pinecone").
        **kwargs: Additional configuration for the store.

    Returns:
        VectorStore instance.
    """
    if backend == "faiss":
        from app.rag.vectorstores.faiss_store import FAISSStore
        return FAISSStore(**kwargs)
    elif backend == "chroma":
        from app.rag.vectorstores.chroma_store import ChromaStore
        return ChromaStore(**kwargs)
    elif backend == "pinecone":
        from app.rag.vectorstores.pinecone_store import PineconeStore
        return PineconeStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")