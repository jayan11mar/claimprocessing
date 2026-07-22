from fastapi.testclient import TestClient

from app.api import server
from app.models.faq import FAQIntent, FAQResponse


def test_chat_includes_retrieval_trace_and_citations_when_present(monkeypatch):
    client = TestClient(server.app)

    def fake_invoke(session_id, user_message, context=None):
        return FAQResponse(
            intent=FAQIntent.OTHER,
            category="faq",
            confidence=0.9,
            answer_text="Answer with evidence",
            reasoning="retrieval backed",
            metadata={
                "retrieval_trace": [{"tool": "knowledge_retrieval", "query": "coverage"}],
                "citations": [{"source_id": "doc-1", "source_path": "policy.md", "text": "Coverage details"}],
            },
        )

    monkeypatch.setattr(server, "_invoke_with_retry", fake_invoke)

    resp = client.post(
        "/chat",
        json={"session_id": "puser4", "message": "What does my policy cover?"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("retrieval_trace") == [{"tool": "knowledge_retrieval", "query": "coverage"}]
    citations = payload.get("citations")
    assert citations == [{"source_id": "doc-1", "source_path": "policy.md", "text": "Coverage details", "source_link": "http://testserver/sources/doc-1/download"}]


def test_ingest_and_retrieve_round_trip():
    client = TestClient(server.app)

    resp = client.post(
        "/ingest",
        json={
            "documents": [
                {
                    "id": "policy-1",
                    "path": "policy.md",
                    "doc_type": "policy_wording",
                    "insurance_type": "health",
                    "content": "Coverage for hospital claims is available after the deductible.",
                }
            ]
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "accepted"
    assert payload["job_id"]

    retrieve_resp = client.post(
        "/retrieve",
        json={"query": "hospital claim coverage", "top_k": 3},
    )

    assert retrieve_resp.status_code == 200
    retrieval_payload = retrieve_resp.json()
    assert retrieval_payload["results"]
    assert retrieval_payload["results"][0]["chunk"]["text"]


def test_health_reports_vector_store_status_and_document_count():
    client = TestClient(server.app)

    resp = client.get("/health")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["vector_store_status"] in {"ok", "empty", "unavailable"}
    assert "document_count" in payload
