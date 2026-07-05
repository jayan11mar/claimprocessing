from app.config import get_settings
from app.rag.chunkers import Chunk
from app.rag.evaluation_harness import evaluate_rag_queries


def test_evaluate_rag_queries_marks_case_passed_when_keywords_and_citations_are_present():
    chunks = [
        Chunk(
            text="Hospitalization coverage applies for inpatient care and related treatment expenses.",
            source_id="policy-1",
            source_path="policy.md",
            doc_type="policy_wording",
            insurance_type="health",
            chunk_index=0,
        )
    ]
    cases = [
        {
            "name": "coverage lookup",
            "query": "Is hospitalization covered?",
            "expected_keywords": ["hospitalization", "coverage"],
            "min_retrieval_score": 0.5,
            "min_answer_score": 0.5,
            "min_citations": 1,
        }
    ]

    def fake_retrieval(chunks_to_search, query, k=3):
        return [
            {
                "chunk": chunks_to_search[0],
                "combined_score": 0.92,
                "source_id": chunks_to_search[0].source_id,
                "source_path": chunks_to_search[0].source_path,
            }
        ]

    def fake_answer(query, chunks=None, claim_context=None, top_k=3, embedding_fn=None):
        return {
            "answer_text": "Hospitalization coverage applies for inpatient care.",
            "citations": [{"source_id": "policy-1", "source_path": "policy.md"}],
            "confidence": 0.91,
        }

    report = evaluate_rag_queries(
        cases=cases,
        chunks=chunks,
        retrieval_fn=fake_retrieval,
        answer_fn=fake_answer,
    )

    assert report["summary"]["passed_cases"] == 1
    assert report["cases"][0]["passed"] is True
    assert report["cases"][0]["retrieval_score"] >= 0.5
    assert report["cases"][0]["answer_score"] >= 0.5


def test_settings_reads_rag_evaluation_thresholds(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RAG_EVALUATION_CONTEXT", "aml / fraud")
    monkeypatch.setenv("RAG_EVALUATION_HIT_RATE_THRESHOLD", "0.8")
    monkeypatch.setenv("RAG_EVALUATION_MRR_THRESHOLD", "0.7")
    monkeypatch.setenv("RAG_EVALUATION_FAITHFULNESS_THRESHOLD", "0.95")
    monkeypatch.setenv("RAG_EVALUATION_ANSWER_CORRECTNESS_THRESHOLD", "0.85")
    monkeypatch.setenv("RAG_EVALUATION_LLM_JUDGE_AVG_THRESHOLD", "4.2")
    monkeypatch.setenv("RAG_EVALUATION_CITATION_COVERAGE_THRESHOLD", "1.0")
    monkeypatch.setenv("RAG_EVALUATION_MIN_CITATIONS", "2")

    settings = get_settings()

    assert settings.RAG_EVALUATION_CONTEXT == "aml / fraud"
    assert settings.RAG_EVALUATION_HIT_RATE_THRESHOLD == 0.8
    assert settings.RAG_EVALUATION_MRR_THRESHOLD == 0.7
    assert settings.RAG_EVALUATION_FAITHFULNESS_THRESHOLD == 0.95
    assert settings.RAG_EVALUATION_ANSWER_CORRECTNESS_THRESHOLD == 0.85
    assert settings.RAG_EVALUATION_LLM_JUDGE_AVG_THRESHOLD == 4.2
    assert settings.RAG_EVALUATION_CITATION_COVERAGE_THRESHOLD == 1.0
    assert settings.RAG_EVALUATION_MIN_CITATIONS == 2

    get_settings.cache_clear()
