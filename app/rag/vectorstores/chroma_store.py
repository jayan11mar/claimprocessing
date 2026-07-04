"""
Chroma vector store implementation for the claims knowledge base.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.api.models import Collection

from app.rag.chunkers import Chunk
from app.rag.vectorstores.base import VectorStore


class ChromaStore(VectorStore):
    """ChromaDB-based vector store implementation."""

    def __init__(
        self,
        collection_name: str = "claims_kb",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize Chroma store.

        Args:
            collection_name: Name of the Chroma collection.
            persist_directory: Directory to persist the database.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(
            Path(__file__).parent.parent.parent.parent / "data" / "chroma_db"
        )
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Collection] = None

    def _get_client(self) -> chromadb.PersistentClient:
        """Get or create the Chroma client."""
        if self._client is None:
            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    def _get_collection(self) -> Collection:
        """Get or create the collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects.
            embeddings: List of embedding vectors corresponding to each chunk.
        """
        collection = self._get_collection()

        ids = [f"{chunk.source_id}_{chunk.chunk_index}" for chunk in chunks]
        documents = [chunk.text for chunk in chunks]

        # Build metadata for each chunk
        metadatas = []
        for chunk in chunks:
            meta = {
                "source_id": chunk.source_id,
                "source_path": chunk.source_path,
                "doc_type": chunk.doc_type,
                "insurance_type": chunk.insurance_type,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.product_code:
                meta["product_code"] = chunk.product_code
            if chunk.product_name:
                meta["product_name"] = chunk.product_name
            if chunk.claim_type:
                meta["claim_type"] = chunk.claim_type
            if chunk.section:
                meta["section"] = chunk.section
            if chunk.clause_id:
                meta["clause_id"] = chunk.clause_id
            # Add raw metadata
            meta.update(chunk.raw_metadata)
            metadatas.append(meta)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

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
        collection = self._get_collection()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunk = Chunk(
                text=results["documents"][0][i],
                source_id=results["metadatas"][0][i].get("source_id", ""),
                source_path=results["metadatas"][0][i].get("source_path", ""),
                doc_type=results["metadatas"][0][i].get("doc_type", ""),
                insurance_type=results["metadatas"][0][i].get("insurance_type", ""),
                product_code=results["metadatas"][0][i].get("product_code"),
                product_name=results["metadatas"][0][i].get("product_name"),
                claim_type=results["metadatas"][0][i].get("claim_type"),
                section=results["metadatas"][0][i].get("section"),
                clause_id=results["metadatas"][0][i].get("clause_id"),
                chunk_index=results["metadatas"][0][i].get("chunk_index", 0),
                raw_metadata={k: v for k, v in results["metadatas"][0][i].items()
                             if k not in ["source_id", "source_path", "doc_type",
                                         "insurance_type", "product_code", "product_name",
                                         "claim_type", "section", "clause_id", "chunk_index"]},
            )
            score = results["distances"][0][i] if "distances" in results else 1.0
            chunks.append((chunk, score))

        return chunks

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """
        Delete entries from the store.

        Args:
            ids: Optional list of source IDs to delete.
                 If None, deletes all entries.
        """
        collection = self._get_collection()

        if ids is None:
            # Delete all - we need to get all IDs first
            all_data = collection.get()
            if all_data["ids"]:
                collection.delete(ids=all_data["ids"])
        else:
            collection.delete(ids=ids)

    def persist(self, path: Optional[str] = None) -> None:
        """
        Persist the store to disk.
        Chroma automatically persists with PersistentClient.

        Args:
            path: Optional path (not used, kept for interface compatibility).
        """
        # Chroma with PersistentClient auto-persists
        # Just ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Return a retriever interface for this store.

        Args:
            search_kwargs: Optional search configuration.

        Returns:
            A retriever object.
        """
        from langchain_chroma import Chroma
        from langchain_core.vectorstores import VectorStore as LangChainVectorStore

        # Create LangChain Chroma wrapper
        langchain_chroma = Chroma(
            client=self._get_client(),
            collection_name=self.collection_name,
        )

        return langchain_chroma.as_retriever(
            search_kwargs=search_kwargs or {"k": 5}
        )

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the store."""
        collection = self._get_collection()
        return collection.count()