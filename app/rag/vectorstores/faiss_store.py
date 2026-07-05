"""
FAISS vector store implementation for the claims knowledge base.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency guard
    faiss = None

import numpy as np

from app.rag.chunkers import Chunk
from app.rag.vectorstores.base import VectorStore


class _FallbackIndex:
    """Small in-memory cosine-similarity index used when faiss is unavailable."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._vectors: List[List[float]] = []
        self._chunk_indices: List[int] = []

    def add(self, embeddings_array: np.ndarray) -> None:
        self._vectors.extend(embeddings_array.tolist())

    def search(self, query_array: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_vector = query_array[0].tolist()
        scores = []
        indices = []
        for idx, vector in enumerate(self._vectors):
            dot = float(np.dot(vector, query_vector))
            scores.append(dot)
            indices.append(idx)

        ranked = sorted(zip(scores, indices), key=lambda item: item[0], reverse=True)
        top_scores = [score for score, _ in ranked[:k]]
        top_indices = [idx for _, idx in ranked[:k]]
        return np.array([top_scores], dtype=np.float32), np.array([top_indices], dtype=np.int64)


class FAISSStore(VectorStore):
    """FAISS-based vector store implementation."""

    def __init__(self, index_path: Optional[str] = None, dimension: Optional[int] = None):
        """
        Initialize FAISS store.

        Args:
            index_path: Path to persist/load the FAISS index.
            dimension: Embedding dimension. If None, inferred from first batch of embeddings.
        """
        self.dimension = dimension
        self.index_path = index_path or str(Path(__file__).parent.parent.parent.parent / "data" / "faiss_index")
        self.index: Optional[faiss.Index] = None
        self._chunks: List[Chunk] = []
        self._chunk_ids: List[str] = []

    def _init_index(self, dimension: int) -> None:
        """Initialize the FAISS index if not already done."""
        if self.index is None:
            self.dimension = dimension
            if faiss is None:
                self.index = _FallbackIndex(dimension)
            else:
                self.index = faiss.IndexFlatIP(dimension)

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects.
            embeddings: List of embedding vectors corresponding to each chunk.
        """
        if not embeddings:
            return
        # Infer dimension from the first embedding if not already set
        inferred_dim = len(embeddings[0])
        self._init_index(inferred_dim)

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        if faiss is not None:
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
        if faiss is not None:
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

        if faiss is None:
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
        search_kwargs = dict(search_kwargs or {})
        embedding_fn = search_kwargs.pop("embedding_fn", None)
        if embedding_fn is None:
            embedding_fn = getattr(self, "embedding_fn", None)
        if embedding_fn is None:
            embedding_fn = getattr(self, "_embedding_fn", None)

        k = search_kwargs.pop("k", 5)
        filter_value = search_kwargs.pop("filter", None)

        from langchain_core.documents import Document as LangChainDocument

        class FAISSRetriever:
            def __init__(self, store: "FAISSStore", k: int = 5, filter: Optional[Dict] = None, embedding_fn: Optional[Any] = None):
                self._store = store
                self._k = k
                self._filter = filter
                self._embedding_fn = embedding_fn

            def invoke(self, input: Any, config: Optional[Dict] = None) -> List[Any]:
                if isinstance(input, dict):
                    query_text = input.get("query") or input.get("text") or ""
                else:
                    query_text = str(input)

                effective_filter = self._filter
                effective_k = self._k
                effective_embedding_fn = self._embedding_fn

                if config is not None:
                    if isinstance(config, dict):
                        effective_filter = config.get("filter", effective_filter)
                        effective_k = config.get("k", effective_k)
                        effective_embedding_fn = config.get("embedding_fn", effective_embedding_fn)

                if effective_embedding_fn is None:
                    return []

                embedding = effective_embedding_fn([query_text])[0]
                results = self._store.search(
                    query=query_text,
                    query_embedding=embedding,
                    k=effective_k,
                    filter=effective_filter,
                )

                documents = []
                for chunk, _score in results:
                    metadata = {
                        "source_id": chunk.source_id,
                        "source_path": chunk.source_path,
                        "doc_type": chunk.doc_type,
                        "insurance_type": chunk.insurance_type,
                        "product_code": chunk.product_code,
                        "product_name": chunk.product_name,
                        "claim_type": chunk.claim_type,
                        "section": chunk.section,
                        "clause_id": chunk.clause_id,
                        "chunk_index": chunk.chunk_index,
                    }
                    metadata.update(chunk.raw_metadata)
                    documents.append(LangChainDocument(page_content=chunk.text, metadata=metadata))

                return documents

            def get_relevant_documents(self, query: Any, config: Optional[Dict] = None) -> List[Any]:
                return self.invoke(query, config=config)

        return FAISSRetriever(self, k=k, filter=filter_value, embedding_fn=embedding_fn)

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the store."""
        return len(self._chunks)