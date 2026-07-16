from fastapi.testclient import TestClient

from app.api.server import app
import app.langsmith_integration as lsi


class FakeMemory:
    def append_message(self, session_id, role, message):
        pass


class FakeAgentChain:
    def invoke(self, session_id, message, context=None):
        if isinstance(context, dict) and isinstance(context.get("timings"), dict):
            context["timings"].update({"llm_ms": 5, "tools": []})
        from app.models.faq import FAQResponse, FAQIntent

        return FAQResponse(
            intent=FAQIntent.OTHER,
            category="test",
            confidence=0.5,
            answer_text="ok",
            reasoning=None,
            metadata={"timings": {"llm_ms": 5, "tools": []}},
        )


def test_chat_sets_langsmith_trace_id_when_sdk_enabled(monkeypatch):
    class FakeRun:
        def __init__(self, id):
            self.id = id

    class FakeClient:
        def create_run(self, name=None, run_type=None, start_time=None, extra=None):
            return FakeRun(id=f"run-{name}")

        def update_run(self, run_id, end_time=None):
            pass

    lsi._client = FakeClient()
    lsi._enabled = True

    from app.chains.agent_chain import AgentChain
    from app.models.faq import FAQResponse, FAQIntent

    agent = AgentChain(memory=FakeMemory())
    agent.faq_chain.invoke = lambda sid, msg, persist_history=True: FAQResponse(
        intent=FAQIntent.OTHER,
        category="test",
        confidence=0.5,
        answer_text="ok",
        reasoning=None,
        metadata={"timings": {"llm_ms": 5, "tools": []}},
    )

    server._memory = FakeMemory()
    server._agent_chain = agent

    client = TestClient(server.app)
    resp = client.post("/chat", json={"session_id": "s-enabled", "message": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert "chain_metadata" in body
    assert body["chain_metadata"].get("langsmith_trace_id") is not None
    assert body["chain_metadata"]["langsmith_trace_id"].startswith("run-agent_invoke:")

    lsi._client = None
    lsi._enabled = False