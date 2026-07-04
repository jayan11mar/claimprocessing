"""
FAISS vector store implementation for the claims knowledge base.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.rag.chunkers import Chunk
from app.rag.vectorstores.base import VectorStore


class FAISSStore(VectorStore):
    """FAISS-based vector store implementation."""

    def __init__(self, index_path: Optional[str] = None, dimension: int = 1536):
        """
        Initialize FAISS store.

        Args:
            index_path: Path to persist/load the FAISS index.
            dimension: Embedding dimension (default 1536 for text-embedding-3-small).
        """
        self.dimension = dimension
        self.index_path = index_path or str(Path(__file__).parent.parent.parent.parent / "data" / "faiss_index")
        self.index: Optional[faiss.Index] = None
        self._chunks: List[Chunk] = []
        self._chunk_ids: List[str] = []

    def _init_index(self) -> None:
        """Initialize the FAISS index if not already done."""
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects.
            embeddings: List of embedding vectors corresponding to each chunk.
        """
        self._init_index()

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity (Inner Product)
        faiss.normalize_L2(embeddings_array)

        # Add to index
        self.index.add(embeddings_array)

        # Store chunks and IDs
        for chunk in chunks:
            self._chunks.append(chunk)
            self._chunk_ids.append(f"{chunk.source_id}_{chunk.chunk_index}")

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
        if self.index is None or len(self._chunks) == 0:
            return []

        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)

        # Search
        scores, indices = self.index.search(query_array, min(k, len(self._chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._chunks):
                chunk = self._chunks[idx]
                # Apply filter if provided
                if filter:
                    match = True
                    for key, value in filter.items():
                        if hasattr(chunk, key):
                            if getattr(chunk, key) != value:
                                match = False
                                break
                        elif key in chunk.raw_metadata:
                            if chunk.raw_metadata[key] != value:
                                match = False
                                break
                        else:
                            match = False
                            break
                    if match:
                        results.append((chunk, float(score)))
                else:
                    results.append((chunk, float(score)))

        return results

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """
        Delete entries from the store.

        Args:
            ids: Optional list of source IDs to delete.
                 If None, deletes all entries.
        """
        if ids is None:
            self.index = None
            self._chunks = []
            self._chunk_ids = []
        else:
            # For FAISS, we need to rebuild the index without the deleted items
            # This is a limitation of FAISS - it doesn't support efficient deletion
            # In production, you might want to use a different approach
            new_chunks = []
            new_ids = []
            for chunk, chunk_id in zip(self._chunks, self._chunk_ids):
                if chunk_id not in ids:
                    new_chunks.append(chunk)
                    new_ids.append(chunk_id)

            self._chunks = new_chunks
            self._chunk_ids = new_ids

            # Rebuild index
            if self._chunks:
                # Re-embed all remaining chunks (in production, you'd cache embeddings)
                # For now, we just mark that index needs rebuild
                self.index = None

    def persist(self, path: Optional[str] = None) -> None:
        """
        Persist the store to disk.

        Args:
            path: Optional path to persist to.
        """
        if self.index is None:
            return

        persist_path = path or self.index_path
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        faiss.write_index(self.index, persist_path)

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Return a retriever interface for this store.

        Args:
            search_kwargs: Optional search configuration.

        Returns:
            A retriever object.
        """
        from langchain_core.retrievers import RetrieverLike
        from langchain_core.vectorstores import VectorStore as LangChainVectorStore

        # Create a simple retriever wrapper
        class FAISSRetriever(RetrieverLike):
            def __init__(self, store: "FAISSStore", k: int = 5, filter: Optional[Dict] = None):
                self._store = store
                self._k = k
                self._filter = filter

            def invoke(self, input: Any, config: Optional[Dict] = None) -> List[Any]:
                # This would need the embedding function to be passed
                # For now, return empty - the retriever_basic will handle this
                return []

        return FAISSRetriever(self, **(search_kwargs or {}))

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the store."""
        return len(self._chunks)