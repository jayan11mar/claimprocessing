""" RAG regression tests.

Technical context: hybrid retrieval, reranking, and streaming QA for claims knowledge retrieval.
"""

from app.rag.chunkers import Chunk
from app.rag.query_transform import build_query_variants
from app.rag.qa_chain import run_qa_chain, stream_qa_chain
from app.rag.reranker import rerank_results
from app.rag.retriever_bm25 import bm25_retrieve
from app.rag.retriever_hybrid import hybrid_retrieve


def make_chunks():
    return [
        Chunk(
            text="Policy exclusions include pre-existing conditions and cosmetic damage.",
            source_id="policy-1",
            source_path="policies/abc.md",
            doc_type="policy_wording",
            insurance_type="auto",
            chunk_index=0,
        ),
        Chunk(
            text="Claims for accidental damage are covered when reported within 30 days.",
            source_id="policy-2",
            source_path="policies/abc.md",
            doc_type="policy_wording",
            insurance_type="auto",
            chunk_index=1,
        ),
        Chunk(
            text="Rejection letters must cite the relevant clause and the evidence used.",
            source_id="letter-1",
            source_path="adjudication_memos/guide.md",
            doc_type="adjudication_memo",
            insurance_type="auto",
            chunk_index=2,
        ),
    ]


def test_bm25_retrieve_prefers_lexical_matches():
    chunks = make_chunks()
    results = bm25_retrieve(chunks, "What exclusions apply to pre-existing conditions?", k=3)

    assert results
    assert results[0].chunk.text.lower().startswith("policy exclusions")


def test_hybrid_retrieve_uses_query_expansion_and_returns_scores():
    chunks = make_chunks()
    variants = build_query_variants("What exclusions apply to pre-existing conditions?")
    assert len(variants) >= 2

    results = hybrid_retrieve(chunks, "What exclusions apply to pre-existing conditions?", k=3)
    assert results
    assert any(result["chunk_id"] for result in results)
    assert any(result["combined_score"] >= 0 for result in results)


def test_reranker_can_reorder_results_with_fallback():
    chunks = make_chunks()
    base_results = [
        {"chunk_id": f"{chunk.source_id}_{chunk.chunk_index}", "chunk": chunk, "combined_score": 0.2}
        for chunk in chunks
    ]
    reranked = rerank_results("What exclusions apply to pre-existing conditions?", base_results, top_k=2)
    assert reranked
    assert len(reranked) <= 2
    assert reranked[0]["chunk_id"]


def test_run_qa_chain_returns_answer_with_citations_and_supports_streaming():
    chunks = make_chunks()
    result = run_qa_chain(
        "What exclusions apply to pre-existing conditions?",
        chunks=chunks,
        top_k=3,
    )

    assert result["answer_text"]
    assert result["citations"]
    assert all("chunk_id" in citation for citation in result["citations"])
    assert all("text" in citation for citation in result["citations"])
    assert result["confidence"] >= 0.0

    streamed = "".join(stream_qa_chain("What exclusions apply to pre-existing conditions?", chunks=chunks, top_k=3))
    assert streamed
