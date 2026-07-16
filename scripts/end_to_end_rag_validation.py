"""End-to-end RAG validation script.

Runs queries through the actual application retrieval path and captures
detailed metrics for validation.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag.retriever_hybrid import hybrid_retrieve
from app.rag.vectorstores import get_vector_store
from app.rag.embeddings import get_embedding_fn
from app.rag.chunkers import Chunk, ChunkConfig, chunk_document
from app.rag.loaders import load_documents_from_manifest

# Configure logging to capture retrieval logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_retrieval_system():
    """Initialize the retrieval system by loading documents and vector store."""
    logger.info("Initializing retrieval system...")
    
    # Load documents from manifest
    documents = load_documents_from_manifest()
    logger.info(f"Loaded {len(documents)} documents from manifest")
    
    # Chunk documents
    config = ChunkConfig(chunk_size=800, chunk_overlap=100)
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, config, use_semantic=True)
        all_chunks.extend(chunks)
    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
    # Load vector store
    settings = get_settings()
    vector_store = get_vector_store(backend=settings.VECTOR_BACKEND)
    if vector_store.index is not None and vector_store.chunk_count > 0:
        logger.info(f"Loaded vector store with {vector_store.chunk_count} chunks")
    else:
        logger.warning("Vector store is empty or not found")
    
    return all_chunks, vector_store, documents


def run_query(
    query: str,
    chunks: List[Chunk],
    vector_store: Any,
    embedding_fn: Any,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run a single query through the hybrid retrieval system.
    
    Returns detailed metrics about the retrieval process.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running query: {query}")
    logger.info(f"{'='*80}")
    
    # Infer insurance type (simulating what agent_chain does)
    from app.chains.agent_chain import AgentChain
    from app.memory.sqlite_memory import SQLiteMemory
    
    memory = SQLiteMemory()
    agent_chain = AgentChain(memory=memory)
    insurance_type = agent_chain._infer_insurance_type(query)
    metadata_filter = {"insurance_type": insurance_type} if insurance_type else None
    
    logger.info(f"Detected insurance_type: {insurance_type}")
    logger.info(f"Metadata filter: {metadata_filter}")
    
    # Run hybrid retrieval
    results = hybrid_retrieve(
        chunks=chunks,
        query=query,
        k=top_k,
        embedding_fn=embedding_fn,
        rerank=True,
        metadata_filter=metadata_filter,
        vector_store=vector_store,
    )
    
    # Extract metrics
    result_count = len(results)
    fallback_used = any(r.get("fallback_used", False) for r in results)
    reranker_used = any(r.get("reranker") == "cross-encoder" for r in results)
    
    # Get BM25 and dense counts (we'll estimate from the results)
    # In the actual implementation, these are internal to hybrid_retrieve
    bm25_count = len([r for r in results if r.get("bm25_score", 0) > 0])
    dense_count = len([r for r in results if r.get("dense_score", 0) > 0])
    
    # Extract top 5 chunks
    top_chunks = []
    for i, result in enumerate(results[:5], 1):
        chunk = result.get("chunk")
        top_chunks.append({
            "rank": i,
            "chunk_id": result.get("chunk_id"),
            "source_id": result.get("source_id"),
            "source_path": result.get("source_path"),
            "doc_type": chunk.doc_type if chunk else None,
            "insurance_type": chunk.insurance_type if chunk else None,
            "text_preview": chunk.text[:200] + "..." if chunk and len(chunk.text) > 200 else (chunk.text if chunk else ""),
            "bm25_score": result.get("bm25_score"),
            "dense_score": result.get("dense_score"),
            "combined_score": result.get("combined_score"),
            "rerank_score": result.get("rerank_score"),
        })
    
    # Build response
    response = {
        "query": query,
        "detected_intent": "KNOWLEDGE_RETRIEVAL",
        "detected_insurance_type": insurance_type,
        "metadata_filter": metadata_filter,
        "fallback_used": fallback_used,
        "retriever_mode": "hybrid" if vector_store and vector_store.chunk_count > 0 else "bm25_only",
        "bm25_result_count": bm25_count,
        "dense_faiss_result_count": dense_count,
        "final_merged_result_count": result_count,
        "reranker_used": reranker_used,
        "top_5_retrieved_chunks": top_chunks,
        "citations": [
            {
                "chunk_id": r.get("chunk_id"),
                "source_id": r.get("source_id"),
                "source_path": r.get("source_path"),
                "text": r.get("chunk").text[:300] if r.get("chunk") else "",
            }
            for r in results[:3]
        ],
    }
    
    logger.info(f"Results: {result_count} chunks retrieved")
    logger.info(f"Fallback used: {fallback_used}")
    logger.info(f"Reranker used: {reranker_used}")
    
    return response


def main():
    """Run all validation queries and generate report."""
    logger.info("Starting end-to-end RAG validation...")
    
    # Initialize system
    chunks, vector_store, documents = initialize_retrieval_system()
    
    # Get embedding function
    settings = get_settings()
    embedding_model = settings.EMBEDDING_MODEL or settings.OPENAI_EMBEDDING_MODEL
    embedding_fn = get_embedding_fn(embedding_model)
    
    # Define test queries
    test_queries = [
        "What is covered under day care procedures in this policy?",
        "What is covered under hospitalization?",
        "What is covered under own damage?",
        "What are the exclusions in this policy?",
        "What documents are required for a health insurance claim?",
    ]
    
    # Run all queries
    results = []
    for query in test_queries:
        try:
            result = run_query(query, chunks, vector_store, embedding_fn, top_k=5)
            results.append(result)
        except Exception as e:
            logger.error(f"Error running query '{query}': {e}", exc_info=True)
            results.append({
                "query": query,
                "error": str(e),
                "detected_intent": "ERROR",
                "detected_insurance_type": None,
                "metadata_filter": None,
                "fallback_used": False,
                "retriever_mode": "error",
                "bm25_result_count": 0,
                "dense_faiss_result_count": 0,
                "final_merged_result_count": 0,
                "reranker_used": False,
                "top_5_retrieved_chunks": [],
                "citations": [],
            })
    
    # Generate validation report
    report = {
        "validation_metadata": {
            "total_queries": len(test_queries),
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "vector_store_chunks": vector_store.chunk_count if vector_store else 0,
            "embedding_model": embedding_model,
        },
        "queries": results,
        "summary": {
            "total_queries": len(test_queries),
            "successful_queries": len([r for r in results if "error" not in r]),
            "failed_queries": len([r for r in results if "error" in r]),
            "queries_with_results": len([r for r in results if r.get("final_merged_result_count", 0) > 0]),
            "queries_with_fallback": len([r for r in results if r.get("fallback_used")]),
            "queries_with_reranker": len([r for r in results if r.get("reranker_used")]),
        }
    }
    
    # Save report
    output_path = ROOT / "END_TO_END_RAG_VALIDATION.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\nValidation report saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("END-TO-END RAG VALIDATION SUMMARY")
    print("="*80)
    print(f"Total queries: {report['summary']['total_queries']}")
    print(f"Successful: {report['summary']['successful_queries']}")
    print(f"Failed: {report['summary']['failed_queries']}")
    print(f"Queries with results: {report['summary']['queries_with_results']}")
    print(f"Queries with fallback: {report['summary']['queries_with_fallback']}")
    print(f"Queries with reranker: {report['summary']['queries_with_reranker']}")
    print("="*80)
    
    # Print detailed results
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}: {result['query']}")
        print(f"  Insurance Type: {result.get('detected_insurance_type')}")
        print(f"  Metadata Filter: {result.get('metadata_filter')}")
        print(f"  Fallback Used: {result.get('fallback_used')}")
        print(f"  Retriever Mode: {result.get('retriever_mode')}")
        print(f"  BM25 Count: {result.get('bm25_result_count')}")
        print(f"  Dense FAISS Count: {result.get('dense_faiss_result_count')}")
        print(f"  Final Result Count: {result.get('final_merged_result_count')}")
        print(f"  Reranker Used: {result.get('reranker_used')}")
        if result.get('top_5_retrieved_chunks'):
            print(f"  Top Chunk: {result['top_5_retrieved_chunks'][0].get('source_id')}")
    
    print("\n" + "="*80)
    
    return report


if __name__ == "__main__":
    report = main()
    sys.exit(0 if report["summary"]["failed_queries"] == 0 else 1)