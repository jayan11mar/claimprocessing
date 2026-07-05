# Vector stores module
from app.rag.vectorstores.base import VectorStore, get_vector_store

__all__ = ["VectorStore", "FAISSStore", "ChromaStore", "PineconeStore", "get_vector_store"]


def __getattr__(name):
    if name == "FAISSStore":
        from app.rag.vectorstores.faiss_store import FAISSStore
        return FAISSStore
    if name == "ChromaStore":
        from app.rag.vectorstores.chroma_store import ChromaStore
        return ChromaStore
    if name == "PineconeStore":
        from app.rag.vectorstores.pinecone_store import PineconeStore
        return PineconeStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
