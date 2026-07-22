from app.chains import router
from app.chains import agent_chain as agent_chain_module
from app.chains.agent_chain import AgentChain
from app.models.faq import FAQIntent, FAQResponse


class _FakeFAQChain:
    def invoke(self, session_id, user_message, persist_history=False):
        return FAQResponse(
            intent=FAQIntent.OTHER,
            category="faq",
            confidence=0.8,
            answer_text="",
            reasoning="generic fallback",
        )


def test_classify_intent_routes_policy_document_queries_to_rag(monkeypatch):
    monkeypatch.setattr(router, "FAQChain", lambda: _FakeFAQChain())

    inputs = {
        "session_id": "sess-1",
        "user_message": "What are the coverages excluded in HDFC Ergo for senior citizen policy?",
    }

    result = router._classify_intent(inputs)

    assert result["_resolved_intent"] == FAQIntent.KNOWLEDGE_RETRIEVAL.value


def test_agent_chain_uses_knowledge_retrieval_for_policy_document_queries(monkeypatch):
    class FakeMemory:
        def append_message(self, *args, **kwargs):
            return None

        def get_history(self, session_id):
            return []

    class FakeFAQChain:
        def __init__(self, memory=None):
            self.memory = memory

        def invoke(self, session_id, user_message, persist_history=False):
            return FAQResponse(
                intent=FAQIntent.OTHER,
                category="faq",
                confidence=0.8,
                answer_text="",
                reasoning="generic fallback",
            )

    monkeypatch.setattr(agent_chain_module, "FAQChain", FakeFAQChain)

    captured = {}

    def fake_knowledge_retrieval(query, top_k=3, claim_context=None, metadata_filter=None):
        captured["query"] = query
        return {
            "answer_text": "Policy exclusions include pre-existing conditions.",
            "citations": [{"source_id": "policy-doc", "source_path": "policy.md", "text": "Exclusions apply."}],
            "confidence": 0.9,
            "retrieval_trace": [{"tool": "knowledge_retrieval", "query": query}],
        }

    monkeypatch.setattr(agent_chain_module, "knowledge_retrieval", fake_knowledge_retrieval, raising=False)

    chain = AgentChain(memory=FakeMemory())
    response = chain.invoke("sess-2", "What are the coverages excluded in HDFC Ergo for senior citizen policy?")

    assert captured["query"] == "What are the coverages excluded in HDFC Ergo for senior citizen policy?"
    assert response.intent == FAQIntent.KNOWLEDGE_RETRIEVAL
    assert response.metadata["citations"]
