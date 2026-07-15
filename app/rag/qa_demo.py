"""CLI for local QA-chain testing."""

import sys
from typing import Optional

from app.config import get_settings
from app.rag.embeddings import get_embedding_fn
from app.rag.qa_chain import run_qa_chain, stream_qa_chain
from app.rag.retriever_hybrid import hybrid_retrieve
from app.rag.chunkers import ChunkConfig, chunk_document
from app.rag.loaders import load_documents_from_manifest


def main(query: Optional[str] = None) -> None:
    if query is None:
        query = sys.stdin.read().strip() or input("Enter a query: ").strip()

    settings = get_settings()
    embedding_model = settings.EMBEDDING_MODEL or settings.OPENAI_EMBEDDING_MODEL
    embedding_fn = get_embedding_fn(embedding_model)

    documents = load_documents_from_manifest()
    chunks = []
    for doc in documents:
        chunks.extend(chunk_document(doc, ChunkConfig(chunk_size=800, chunk_overlap=100), use_semantic=True))

    results = hybrid_retrieve(chunks, query, k=5, embedding_fn=embedding_fn)
    qa_result = run_qa_chain(query, chunks=chunks, top_k=5, embedding_fn=embedding_fn)

    print("ANSWER")
    for chunk in stream_qa_chain(query, chunks=chunks, top_k=5):
        print(chunk, end="", flush=True)
    print()
    print("\nCITATIONS")
    for citation in qa_result["citations"]:
        rerank_score = citation.get("rerank_score", "N/A")
        print(f"- {citation['chunk_id']}: {citation['source_path']} (rerank_score={rerank_score})")
    print("\nTOP 5 RETRIEVED CHUNKS (post-rerank)")
    for result in results:
        rerank_s = result.get("rerank_score", "N/A")
        print(f"- {result['chunk_id']} (combined={result['combined_score']}, rerank={rerank_s})")
        print(result["chunk"].text[:220])
        print()


if __name__ == "__main__":
    main()
