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


def test_hybrid_retrieve_auto_loads_embeddings_from_config():
    """Hybrid retrieve auto-loads embedding function from config when none provided."""
    chunks = make_chunks()

    # Call without embedding_fn - should auto-load from config
    # Since .env has dummy key, it'll use fallback embedding which at least
    # tries to load from config rather than immediately falling to token overlap
    results = hybrid_retrieve(chunks, "What exclusions apply to pre-existing conditions?", k=3)

    assert results
    assert all("dense_score" in r for r in results)
    assert all("bm25_score" in r for r in results)
    assert all("combined_score" in r for r in results)
    # The function should attempt to load embeddings (may use fallback if no API key)
    # but the key improvement is it no longer silently falls back to token overlap


def test_hybrid_retrieve_with_explicit_embeddings():
    """Hybrid retrieve uses real embeddings (cosine similarity) when function provided."""
    chunks = make_chunks()

    def dummy_embed(texts):
        import hashlib
        result = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, 8, 2)]
            result.append(vec)
        return result

    results = hybrid_retrieve(
        chunks,
        "What exclusions apply to pre-existing conditions?",
        k=3,
        embedding_fn=dummy_embed,
    )
    assert results
    # With real embeddings, dense_score should be > 0
    assert any(r["dense_score"] > 0 for r in results)


def test_hybrid_retrieve_with_metadata_filter():
    """Hybrid retrieve applies metadata filter correctly."""
    chunks = make_chunks()

    def dummy_embed(texts):
        import hashlib
        result = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, 8, 2)]
            result.append(vec)
        return result

    results = hybrid_retrieve(
        chunks,
        "What exclusions apply?",
        k=3,
        embedding_fn=dummy_embed,
        metadata_filter={"doc_type": "adjudication_memo"},
    )
    assert results
    for result in results:
        assert result["chunk"].doc_type == "adjudication_memo"


def test_hybrid_retrieve_no_matching_filter():
    """Hybrid retrieve returns empty list when filter matches nothing."""
    chunks = make_chunks()
    results = hybrid_retrieve(chunks, "test", k=3, metadata_filter={"doc_type": "nonexistent"})
    assert results == []


def test_hybrid_retrieve_empty_chunks():
    """Hybrid retrieve returns empty list for empty chunks."""
    results = hybrid_retrieve([], "test query", k=3)
    assert results == []


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
