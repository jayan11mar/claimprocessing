"""
Pinecone vector store implementation for the claims knowledge base.
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:  # pragma: no cover - optional dependency guard
    Pinecone = None
    ServerlessSpec = None

from app.rag.chunkers import Chunk
from app.rag.metadata import build_chunk_metadata
from app.rag.vectorstores.base import VectorStore


class PineconeStore(VectorStore):
    """Pinecone-based vector store implementation."""

    def __init__(
        self,
        index_name: str = "claims-kb",
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        dimension: int = 1536,
    ):
        """
        Initialize Pinecone store.

        Args:
            index_name: Name of the Pinecone index.
            api_key: Pinecone API key.
            environment: Pinecone environment.
            dimension: Embedding dimension.
        """
        self.index_name = index_name
        self.dimension = dimension
        self._api_key = api_key
        self.environment = environment
        self._pc: Optional[Any] = None
        self._index: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create the Pinecone client."""
        if Pinecone is None:
            raise RuntimeError("pinecone is not installed; install it to use the Pinecone backend")
        if self._pc is None:
            from app.config import get_settings
            settings = get_settings()
            api_key = self._api_key or getattr(settings, "PINECONE_API_KEY", None)
            if not api_key:
                raise RuntimeError("PINECONE_API_KEY is not configured")
            self._pc = Pinecone(api_key=api_key)
        return self._pc

    def _get_index(self) -> Any:
        """Get or create the index."""
        if self._index is None:
            if ServerlessSpec is None:
                raise RuntimeError("pinecone serverless spec is unavailable")
            pc = self._get_client()
            # Check if index exists, create if not
            try:
                pc.describe_index(self.index_name)
            except Exception:
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            self._index = pc.Index(self.index_name)
        return self._index

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects.
            embeddings: List of embedding vectors corresponding to each chunk.
        """
        index = self._get_index()

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = build_chunk_metadata(chunk)
            metadata["source_id"] = chunk.source_id
            metadata["source_path"] = chunk.source_path
            metadata["chunk_index"] = chunk.chunk_index

            vectors.append({
                "id": f"{chunk.source_id}_{chunk.chunk_index}",
                "values": embedding,
                "metadata": metadata,
            })

        index.upsert(vectors=vectors)

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
        index = self._get_index()

        results = index.query(
            vector=query_embedding,
            top_k=k,
            filter=filter,
            include_metadata=True,
        )

        chunks = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            chunk = Chunk(
                text=metadata.get("text", ""),
                source_id=metadata.get("source_id", ""),
                source_path=metadata.get("source_path", ""),
                doc_type=metadata.get("doc_type", ""),
                insurance_type=metadata.get("insurance_type", ""),
                insurer=metadata.get("insurer"),
                product_code=metadata.get("product_code"),
                product_name=metadata.get("product_name"),
                claim_type=metadata.get("claim_type"),
                section=metadata.get("section"),
                clause_id=metadata.get("clause_id"),
                chunk_index=metadata.get("chunk_index", 0),
                raw_metadata={k: v for k, v in metadata.items()
                             if k not in ["source_id", "source_path", "doc_type",
                                         "insurance_type", "insurer", "product_code", "product_name",
                                         "claim_type", "section", "clause_id", "chunk_index", "text"]},
            )
            score = match.get("score", 1.0)
            chunks.append((chunk, score))

        return chunks

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """
        Delete entries from the store.

        Args:
            ids: Optional list of source IDs to delete.
                 If None, deletes all entries (not supported in Pinecone).
        """
        index = self._get_index()

        if ids:
            index.delete(ids=ids)
        # Note: Pinecone doesn't support deleting all vectors easily

    def persist(self, path: Optional[str] = None) -> None:
        """
        Persist the store.
        Pinecone is a managed service, so this is a no-op.

        Args:
            path: Optional path (not used).
        """
        pass

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Return a retriever interface for this store.

        Args:
            search_kwargs: Optional search configuration.

        Returns:
            A retriever object.
        """
        # For Pinecone, we'd need to use langchain-pinecone
        # This is a placeholder for the interface
        raise NotImplementedError("Pinecone retriever requires langchain-pinecone integration")

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the store."""
        # Pinecone doesn't have a direct count method
        return -1  # Unknown