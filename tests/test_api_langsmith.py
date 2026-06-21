from fastapi.testclient import TestClient

from app.api.server import app


class FakeMemory:
    def append_message(self, session_id, role, message):
        pass


class FakeAgentChain:
    def invoke(self, session_id, message, context=None):
        if isinstance(context, dict) and isinstance(context.get("timings"), dict):
            context["timings"].update({"llm_ms": 10, "tools": []})
        from app.models.faq import FAQResponse, FAQIntent

        return FAQResponse(
            intent=FAQIntent.OTHER,
            category="test",
            confidence=0.5,
            answer_text="ok",
            reasoning=None,
            metadata={"timings": {"llm_ms": 10, "tools": []}},
        )


def test_chat_includes_langsmith_trace_id(monkeypatch):
    server._memory = FakeMemory()
    server._agent_chain = FakeAgentChain()

    monkeypatch.setattr(server, "get_langsmith_trace_id", lambda: "ls-test-id")

    client = TestClient(server.app)
    resp = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert "chain_metadata" in body
    assert body["chain_metadata"].get("langsmith_trace_id") == "ls-test-id"