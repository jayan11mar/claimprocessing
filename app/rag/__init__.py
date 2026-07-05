# RAG module for claims knowledge base
from app.rag.chunkers import recursive_chunk, semantic_chunk
from app.rag.embeddings import get_embedding_fn
from app.rag.vectorstores import get_vector_store

__all__ = [
    "load_manifest",
    "iter_manifest_sources",
    "load_documents_from_manifest",
    "recursive_chunk",
    "semantic_chunk",
    "get_embedding_fn",
    "get_vector_store",
    "build_basic_retriever",
]


def __getattr__(name):
    if name in {"load_manifest", "iter_manifest_sources", "load_documents_from_manifest"}:
        from app.rag.loaders import load_manifest, iter_manifest_sources, load_documents_from_manifest
        return {
            "load_manifest": load_manifest,
            "iter_manifest_sources": iter_manifest_sources,
            "load_documents_from_manifest": load_documents_from_manifest,
        }[name]
    if name == "build_basic_retriever":
        from app.rag.retriever_basic import build_basic_retriever
        return build_basic_retriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")