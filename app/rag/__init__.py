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
    "bm25_retrieve",
    "hybrid_retrieve",
    "run_qa_chain",
    "stream_qa_chain",
    "rerank_results",
    "evaluate_rag_queries",
    "run_rag_evaluation",
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
    if name == "bm25_retrieve":
        from app.rag.retriever_bm25 import bm25_retrieve
        return bm25_retrieve
    if name == "hybrid_retrieve":
        from app.rag.retriever_hybrid import hybrid_retrieve
        return hybrid_retrieve
    if name == "run_qa_chain":
        from app.rag.qa_chain import run_qa_chain
        return run_qa_chain
    if name == "stream_qa_chain":
        from app.rag.qa_chain import stream_qa_chain
        return stream_qa_chain
    if name == "rerank_results":
        from app.rag.reranker import rerank_results
        return rerank_results
    if name in {"evaluate_rag_queries", "run_rag_evaluation"}:
        from app.rag.evaluation_harness import evaluate_rag_queries, run_rag_evaluation
        return {
            "evaluate_rag_queries": evaluate_rag_queries,
            "run_rag_evaluation": run_rag_evaluation,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")