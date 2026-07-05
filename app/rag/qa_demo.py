"""CLI for local QA-chain testing."""

import sys
from typing import Optional

from app.rag.qa_chain import run_qa_chain, stream_qa_chain
from app.rag.retriever_hybrid import hybrid_retrieve
from app.rag.chunkers import ChunkConfig, chunk_document
from app.rag.loaders import load_documents_from_manifest


def main(query: Optional[str] = None) -> None:
    if query is None:
        query = sys.stdin.read().strip() or input("Enter a query: ").strip()

    documents = load_documents_from_manifest()
    chunks = []
    for doc in documents:
        chunks.extend(chunk_document(doc, ChunkConfig(chunk_size=800, chunk_overlap=100), use_semantic=True))

    results = hybrid_retrieve(chunks, query, k=5)
    qa_result = run_qa_chain(query, chunks=chunks, top_k=5)

    print("ANSWER")
    for chunk in stream_qa_chain(query, chunks=chunks, top_k=5):
        print(chunk, end="", flush=True)
    print()
    print("\nCITATIONS")
    for citation in qa_result["citations"]:
        print(f"- {citation['chunk_id']}: {citation['source_path']}")
    print("\nTOP 5 RETRIEVED CHUNKS")
    for result in results:
        print(f"- {result['chunk_id']} (score={result['combined_score']})")
        print(result["chunk"].text[:220])
        print()


if __name__ == "__main__":
    main()
