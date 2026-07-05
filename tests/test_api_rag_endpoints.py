from fastapi.testclient import TestClient

from app.api import server
from app.models.faq import FAQIntent, FAQResponse


def test_week6_rag_endpoints_support_ingest_retrieve_and_chat_flow():
    client = TestClient(server.app)

    ingest_resp = client.post(
        "/ingest",
        json={
            "documents": [
                {
                    "id": "policy-week6",
                    "path": "policy-week6.md",
                    "doc_type": "policy_wording",
                    "insurance_type": "health",
                    "content": "Hospital claims are covered after the deductible and prior authorization is required for elective procedures.",
                }
            ]
        },
    )

    assert ingest_resp.status_code == 200
    ingest_payload = ingest_resp.json()
    assert ingest_payload["status"] == "accepted"
    assert ingest_payload["job_id"]

    status_resp = client.get(f"/ingest/status/{ingest_payload['job_id']}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] in {"running", "completed", "failed"}

    retrieve_resp = client.post(
        "/retrieve",
        json={"query": "hospital claim deductible", "top_k": 2},
    )

    assert retrieve_resp.status_code == 200
    retrieval_payload = retrieve_resp.json()
    assert retrieval_payload["results"]
    assert retrieval_payload["source_count"] >= 1


def test_week6_chat_response_exposes_retrieval_citations(monkeypatch):
    client = TestClient(server.app)

    def fake_invoke(session_id, user_message, context=None):
        return FAQResponse(
            intent=FAQIntent.OTHER,
            category="faq",
            confidence=0.95,
            answer_text="Coverage details are available in the policy wording.",
            reasoning="retrieval-backed answer",
            metadata={
                "retrieval_trace": [{"tool": "knowledge_retrieval", "query": "coverage"}],
                "citations": [{"source_id": "policy-week6", "source_path": "policy-week6.md", "text": "Hospital claims are covered after the deductible."}],
            },
        )

    monkeypatch.setattr(server, "_invoke_with_retry", fake_invoke)

    chat_resp = client.post(
        "/chat",
        json={"session_id": "week6-session", "message": "What is covered for hospital claims?"},
    )

    assert chat_resp.status_code == 200
    payload = chat_resp.json()
    assert payload["retrieval_trace"]
    assert payload["citations"]
