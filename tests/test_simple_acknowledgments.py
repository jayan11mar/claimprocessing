from app.chains.faq_chain import FAQChain
from app.models.faq import FAQIntent


class FakeMemory:
    def __init__(self):
        self.store = {}

    def append_message(self, session_id, role, message):
        self.store.setdefault(session_id, []).append((role, message))

    def get_history(self, session_id):
        return self.store.get(session_id, [])

    def clear_history(self, session_id):
        self.store.pop(session_id, None)


def test_simple_acknowledgment_ok():
    chain = FAQChain(memory=FakeMemory())
    response = chain.invoke("test-session", "OK", persist_history=False)
    assert response.intent == FAQIntent.OTHER
    assert response.confidence == 1.0
    assert "You're welcome" in response.answer_text
    assert response.metadata.get("simple_acknowledgment") is True


def test_simple_acknowledgment_thank_you():
    chain = FAQChain(memory=FakeMemory())
    response = chain.invoke("test-session", "Thank you", persist_history=False)
    assert response.intent == FAQIntent.OTHER
    assert response.confidence == 1.0
    assert "You're welcome" in response.answer_text
    assert response.metadata.get("simple_acknowledgment") is True


def test_simple_acknowledgment_lowercase():
    chain = FAQChain(memory=FakeMemory())
    response = chain.invoke("test-session", "ok", persist_history=False)
    assert response.intent == FAQIntent.OTHER
    assert response.confidence == 1.0
    assert "You're welcome" in response.answer_text


def test_simple_acknowledgment_thanks():
    chain = FAQChain(memory=FakeMemory())
    response = chain.invoke("test-session", "thanks", persist_history=False)
    assert response.intent == FAQIntent.OTHER
    assert response.confidence == 1.0
    assert "You're welcome" in response.answer_text


def test_greeting_hi():
    chain = FAQChain(memory=FakeMemory())
    response = chain.invoke("test-session", "HI", persist_history=False)
    assert response.intent == FAQIntent.OTHER
    assert response.confidence == 1.0
    assert "Hello" in response.answer_text
    assert response.category == "greeting"
    assert response.metadata.get("simple_acknowledgment") is True


def test_greeting_hello():
    chain = FAQChain(memory=FakeMemory())
    response = chain.invoke("test-session", "Hello", persist_history=False)
    assert response.intent == FAQIntent.OTHER
    assert response.confidence == 1.0
    assert "Hello" in response.answer_text
    assert response.category == "greeting"


def test_greeting_hey():
    chain = FAQChain(memory=FakeMemory())
    response = chain.invoke("test-session", "Hey", persist_history=False)
    assert response.intent == FAQIntent.OTHER
    assert response.confidence == 1.0
    assert "Hi there" in response.answer_text
    assert response.category == "greeting"


def test_non_acknowledgment_passes_through():
    chain = FAQChain(memory=FakeMemory())
    response = chain.invoke("test-session", "What is my policy status?", persist_history=False)
    # Should not be treated as a simple acknowledgment
    assert response.metadata.get("simple_acknowledgment") is not True
