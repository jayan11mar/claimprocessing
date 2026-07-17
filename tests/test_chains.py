"""Tests for the LCEL orchestration layer.

Validates:
1. ``base_lcel`` helpers (``make_retryable``, ``make_fallback_chain``, ``build_run_dict``)
2. ``rag_chain_lcel`` wrapper
3. ``tool_chain_lcel`` wrapper
4. ``hitl_chain`` placeholder
5. ``router`` intent → chain dispatch
6. Callback handlers (logging, tracing, metrics)
7. Smoke test: router returns a valid ``ChatResponse``-shaped dict
"""

import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from app.chains.base_lcel import (
    build_run_dict,
    lcel_identity,
    make_fallback_chain,
    make_retryable,
)
from app.chains.hitl_chain import hitl_lcel_chain
from app.chains.rag_chain_lcel import rag_lcel_chain
from app.chains.router import lcel_router
from app.chains.tool_chain_lcel import tool_lcel_chain
from app.models.faq import FAQIntent, FAQResponse


# =========================================================================
# base_lcel helpers
# =========================================================================


class TestBaseLcel:

    def test_build_run_dict(self):
        result = build_run_dict("sess-1", "hello", {"key": "val"})
        assert result == {"session_id": "sess-1", "user_message": "hello", "metadata": {"key": "val"}}

    def test_build_run_dict_default_metadata(self):
        result = build_run_dict("sess-1", "hello")
        assert result["metadata"] == {}

    def test_lcel_identity_passthrough(self):
        inputs = {"a": 1, "b": 2}
        result = lcel_identity.invoke(inputs)
        assert result == inputs

    def test_make_retryable_success(self):
        """A runnable that succeeds on first call should work normally."""
        runnable = lcel_identity
        retried = make_retryable(runnable, max_retries=2)
        result = retried.invoke({"ok": True})
        assert result == {"ok": True}

    def test_make_fallback_chain_primary_success(self):
        """When primary succeeds, fallback is not used."""
        primary = lcel_identity
        fallback = lcel_identity
        chain = make_fallback_chain(primary, fallback)
        result = chain.invoke({"from": "primary"})
        assert result == {"from": "primary"}


# =========================================================================
# rag_chain_lcel
# =========================================================================


class TestRagLcelChain:

    @patch("app.chains.rag_chain_lcel.knowledge_retrieval")
    def test_rag_chain_invoke(self, mock_kr):
        mock_kr.return_value = {
            "answer_text": "Policy covers hospital claims.",
            "citations": [{"chunk_id": "c1", "text": "Coverage text"}],
            "confidence": 0.92,
            "retrieval_trace": [{"tool": "knowledge_retrieval", "query": "test"}],
        }

        result = rag_lcel_chain.invoke({
            "user_message": "What does the policy cover?",
            "metadata": {"metadata_filter": {"insurance_type": "health"}},
        })

        assert result["answer_text"] == "Policy covers hospital claims."
        assert len(result["citations"]) == 1
        assert result["confidence"] == 0.92
        assert len(result["retrieval_trace"]) == 1

        mock_kr.assert_called_once_with(
            query="What does the policy cover?",
            top_k=3,
            claim_context=None,
            metadata_filter={"insurance_type": "health"},
        )

    @patch("app.chains.rag_chain_lcel.knowledge_retrieval")
    def test_rag_chain_empty_message(self, mock_kr):
        mock_kr.return_value = {
            "answer_text": "",
            "citations": [],
            "confidence": 0.0,
            "retrieval_trace": [],
        }

        result = rag_lcel_chain.invoke({
            "user_message": "",
            "metadata": {},
        })

        assert result["answer_text"] == ""
        assert result["citations"] == []


# =========================================================================
# tool_chain_lcel
# =========================================================================


class TestToolLcelChain:

    @patch("app.chains.tool_chain_lcel.AgentChain")
    @patch("app.chains.tool_chain_lcel.SQLiteMemory")
    def test_tool_chain_invoke(self, mock_memory_cls, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = FAQResponse(
            intent=FAQIntent.POLICY_STATUS,
            category="policy",
            confidence=0.95,
            answer_text="Policy P12345 is active.",
            reasoning="Policy status check",
            metadata={
                "tool": "policy_checker",
                "tool_output": {"policy_number": "P12345", "status": "active"},
            },
        )
        mock_agent_cls.return_value = mock_agent

        result = tool_lcel_chain.invoke({
            "session_id": "sess-1",
            "user_message": "Check policy P12345",
            "metadata": {"correlation_id": "corr-1"},
        })

        assert result["answer_text"] == "Policy P12345 is active."
        assert result["intent"] == "POLICY_STATUS"
        assert result["confidence"] == 0.95
        assert result["tool_metadata"]["tool"] == "policy_checker"
        # is_tool_augmented is False because the mock AgentChain does not
        # populate tool timings; the real AgentChain would set it to True.
        assert result["chain_metadata"]["is_tool_augmented"] is False

    @patch("app.chains.tool_chain_lcel.AgentChain")
    @patch("app.chains.tool_chain_lcel.SQLiteMemory")
    def test_tool_chain_fallback_on_error(self, mock_memory_cls, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_agent_cls.return_value = mock_agent

        with pytest.raises(RuntimeError):
            tool_lcel_chain.invoke({
                "session_id": "sess-err",
                "user_message": "test",
            })


# =========================================================================
# hitl_chain (placeholder)
# =========================================================================


class TestHitlChain:

    def test_hitl_placeholder_skipped(self):
        """When ENABLE_HITL is False, hitl chain returns pass-through with skipped status."""
        import os
        if "ENABLE_HITL" in os.environ:
            del os.environ["ENABLE_HITL"]
        from app.config import get_settings
        get_settings.cache_clear()
        inputs = {
            "session_id": "sess-1",
            "user_message": "Approve claim C1001",
            "metadata": {},
        }
        result = hitl_lcel_chain.invoke(inputs)
        assert result["hitl"]["status"] == "skipped"
        assert result["session_id"] == "sess-1"
        assert result["user_message"] == "Approve claim C1001"

    def test_hitl_chain_pass_through_when_no_trigger(self):
        """When HITL is enabled but no rule matches, chain returns passed status."""
        import os
        os.environ["ENABLE_HITL"] = "true"
        from app.config import get_settings
        get_settings.cache_clear()
        from app.hitl.store import reset_task_store_singleton
        from app.hitl.manager import reset_hitl_manager_singleton
        from app.hitl.triggers import clear_rules_cache
        reset_task_store_singleton()
        reset_hitl_manager_singleton()
        clear_rules_cache()

        inputs = {
            "session_id": "sess-1",
            "user_message": "Approve claim C1001",
            "metadata": {},
        }
        result = hitl_lcel_chain.invoke(inputs)
        assert result["hitl"]["status"] == "passed"
        assert result["hitl_paused"] is False

    def test_hitl_chain_pauses_on_trigger(self):
        """When a trigger rule matches (high amount), chain must pause and return task_id."""
        import os
        import tempfile
        os.environ["ENABLE_HITL"] = "true"
        os.environ["HITL_STORE_PATH"] = tempfile.mktemp(suffix=".db")
        from app.config import get_settings
        get_settings.cache_clear()
        from app.hitl.store import reset_task_store_singleton
        from app.hitl.manager import reset_hitl_manager_singleton
        from app.hitl.triggers import clear_rules_cache
        reset_task_store_singleton()
        reset_hitl_manager_singleton()
        clear_rules_cache()

        inputs = {
            "session_id": "sess-hitl-trigger",
            "user_message": "Process claim for Rs 600,000",
            "claim_amount": 600000,
            "decision": "pending",
            "confidence": 0.72,
            "recommendation": {"action": "manual_review", "amount": 480000},
            "retrieved_chunks": [{"chunk_id": "c1", "text": "Policy limit clause"}],
        }
        result = hitl_lcel_chain.invoke(inputs)
        assert result["hitl"]["status"] == "pending"
        assert result["hitl_paused"] is True
        assert "hitl_task_id" in result
        assert result["hitl_task_id"].startswith("hitl_")


# =========================================================================
# Router
# =========================================================================


class TestRouter:

    @patch("app.chains.router.FAQChain")
    def test_router_returns_dict_with_required_keys(self, mock_faq_cls):
        """Smoke test: router returns a dict with the /chat contract keys."""
        mock_faq = MagicMock()
        mock_faq.invoke.return_value = FAQResponse(
            intent=FAQIntent.OTHER,
            category="general",
            confidence=0.5,
            answer_text="Hello! How can I assist you?",
            reasoning="Simple greeting",
            metadata={},
        )
        mock_faq_cls.return_value = mock_faq

        result = lcel_router.invoke({
            "session_id": "sess-smoke",
            "user_message": "hello",
            "metadata": {},
        })

        # Must have the core /chat contract keys
        assert "answer_text" in result
        assert "intent" in result
        assert "confidence" in result
        assert "metadata" in result
        assert result["answer_text"] is not None

    @patch("app.chains.router.FAQChain")
    @patch("app.chains.rag_chain_lcel.knowledge_retrieval")
    def test_router_rag_routing(self, mock_kr, mock_faq_cls):
        """When FAQChain returns KNOWLEDGE_RETRIEVAL, router should route to RAG chain."""
        mock_faq = MagicMock()
        mock_faq.invoke.return_value = FAQResponse(
            intent=FAQIntent.KNOWLEDGE_RETRIEVAL,
            category="knowledge",
            confidence=0.85,
            answer_text="What does the policy cover?",
            reasoning="Knowledge retrieval intent",
            metadata={},
        )
        mock_faq_cls.return_value = mock_faq

        mock_kr.return_value = {
            "answer_text": "Policy covers hospital claims.",
            "citations": [],
            "confidence": 0.9,
            "retrieval_trace": [],
        }

        result = lcel_router.invoke({
            "session_id": "sess-rag",
            "user_message": "What does the policy cover?",
            "metadata": {},
        })

        assert result["answer_text"] == "Policy covers hospital claims."

    @patch("app.chains.router.FAQChain")
    def test_router_internal_keys_stripped(self, mock_faq_cls):
        """Internal keys (_resolved_intent, _faq_confidence) must not appear in output."""
        mock_faq = MagicMock()
        mock_faq.invoke.return_value = FAQResponse(
            intent=FAQIntent.OTHER,
            category="general",
            confidence=0.5,
            answer_text="Hi",
            reasoning="Greeting",
            metadata={},
        )
        mock_faq_cls.return_value = mock_faq

        result = lcel_router.invoke({
            "session_id": "sess-strip",
            "user_message": "hi",
            "metadata": {},
        })

        assert "_resolved_intent" not in result
        assert "_faq_confidence" not in result


# =========================================================================
# Callback handlers
# =========================================================================


class TestCallbacks:

    def test_logging_callback_handler(self):
        """LoggingCallbackHandler can be instantiated and attached."""
        from app.callbacks.logging_cb import LoggingCallbackHandler
        handler = LoggingCallbackHandler(session_id="sess-1")
        assert handler.session_id == "sess-1"

        # Verify it has the expected lifecycle methods
        assert hasattr(handler, "on_chain_start")
        assert hasattr(handler, "on_chain_end")
        assert hasattr(handler, "on_chain_error")
        assert hasattr(handler, "on_llm_start")
        assert hasattr(handler, "on_llm_end")
        assert hasattr(handler, "on_tool_start")
        assert hasattr(handler, "on_tool_end")

    def test_tracing_callback_handler(self):
        """TracingCallbackHandler can be instantiated and attached."""
        from app.callbacks.tracing_cb import TracingCallbackHandler
        handler = TracingCallbackHandler(session_id="sess-1", trace_id="trace-abc")
        assert handler.session_id == "sess-1"
        assert handler.trace_id == "trace-abc"

    def test_metrics_callback_handler(self):
        """MetricsCallbackHandler collects and reports metrics."""
        from app.callbacks.metrics_cb import MetricsCallbackHandler
        handler = MetricsCallbackHandler()

        # Simulate a chain start/end cycle
        handler.on_chain_start({}, {}, run_id="run-1")
        time.sleep(0.001)
        handler.on_chain_end(run_id="run-1")

        report = handler.report()
        assert "chain" in report
        assert report["chain"]["count"] == 1
        assert report["chain"]["total_ms"] > 0
        assert report["chain"]["errors"] == 0

    def test_metrics_callback_handler_errors(self):
        """MetricsCallbackHandler tracks errors."""
        from app.callbacks.metrics_cb import MetricsCallbackHandler
        handler = MetricsCallbackHandler()

        handler.on_chain_start({}, {}, run_id="run-err")
        handler.on_chain_error(RuntimeError("fail"), run_id="run-err")

        report = handler.report()
        assert report["chain"]["errors"] == 1
        assert report["chain"]["count"] == 1  # error still counts as an invocation


# =========================================================================
# Smoke test: router returns valid ChatResponse-shaped output
# =========================================================================


@pytest.mark.smoke
@patch("app.chains.router.FAQChain")
def test_router_smoke(mock_faq_cls):
    """Smoke test: the router can be invoked and returns a dict with the
    expected /chat contract fields."""
    mock_faq = MagicMock()
    mock_faq.invoke.return_value = FAQResponse(
        intent=FAQIntent.OTHER,
        category="general",
        confidence=0.5,
        answer_text="Hello!",
        reasoning="Greeting",
        metadata={},
    )
    mock_faq_cls.return_value = mock_faq

    result = lcel_router.invoke({
        "session_id": "smoke-test",
        "user_message": "hello",
        "metadata": {},
    })

    # Core contract fields
    assert isinstance(result, dict)
    assert "answer_text" in result
    assert "intent" in result
    assert "confidence" in result
    assert "metadata" in result

    # answer_text must be a non-empty string for a valid greeting
    assert isinstance(result["answer_text"], str)
    assert len(result["answer_text"]) > 0