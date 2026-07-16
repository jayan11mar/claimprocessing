"""Additional tests for app/langsmith_integration.py to achieve >85% coverage.

Covers the branches that the existing test_langsmith_integration_unit.py misses:
- init_langsmith with various missing/partial configs
- start_trace with enabled client
- record_span with various client methods
- Edge cases in _safe_client_call
"""

import os
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from app.langsmith_integration import (
    init_langsmith,
    start_trace,
    record_span,
    get_langsmith_trace_id,
    _enabled,
    _client,
    _last_trace_id,
)


class TestInitLangsmith:
    def test_init_disabled_when_no_api_key(self):
        """init_langsmith returns None when LANGSMITH_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = init_langsmith()
            assert result is None
            from app.langsmith_integration import _enabled
            assert _enabled is False

    def test_init_disabled_when_ls_not_available(self):
        """init_langsmith returns None when langsmith package is not available."""
        with patch.dict(os.environ, {
            "LANGSMITH_API_KEY": "sk-ls-test",
            "LANGSMITH_TRACING": "true",
        }, clear=True):
            with patch("app.langsmith_integration._LS_AVAILABLE", False):
                result = init_langsmith()
                assert result is None

    def test_init_disabled_when_tracing_flag_not_set(self):
        """init_langsmith returns None when LANGSMITH_TRACING is not set."""
        with patch.dict(os.environ, {
            "LANGSMITH_API_KEY": "sk-ls-test",
        }, clear=True):
            result = init_langsmith()
            assert result is None

    def test_init_disabled_when_tracing_flag_is_false(self):
        """init_langsmith returns None when LANGSMITH_TRACING is 'false'."""
        with patch.dict(os.environ, {
            "LANGSMITH_API_KEY": "sk-ls-test",
            "LANGSMITH_TRACING": "false",
        }, clear=True):
            result = init_langsmith()
            assert result is None

    def test_init_creates_client_successfully(self):
        """init_langsmith creates a client and enables tracing."""
        with patch.dict(os.environ, {
            "LANGSMITH_API_KEY": "sk-ls-test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT_NAME": "test-project",
        }, clear=True):
            mock_client = MagicMock()
            with patch("app.langsmith_integration.LangSmithClient", return_value=mock_client):
                result = init_langsmith()
                assert result is mock_client
                from app.langsmith_integration import _enabled
                assert _enabled is True

    def test_init_fallback_client_creation(self):
        """init_langsmith tries fallback client creation on TypeError."""
        with patch.dict(os.environ, {
            "LANGSMITH_API_KEY": "sk-ls-test",
            "LANGSMITH_TRACING": "true",
        }, clear=True):
            # First call raises TypeError, second succeeds
            mock_client = MagicMock()
            with patch("app.langsmith_integration.LangSmithClient") as mock_ls:
                mock_ls.side_effect = [TypeError("bad"), mock_client]
                result = init_langsmith()
                assert result is mock_client

    def test_init_all_fallbacks_fail(self):
        """init_langsmith returns None when all client creation attempts fail."""
        with patch.dict(os.environ, {
            "LANGSMITH_API_KEY": "sk-ls-test",
            "LANGSMITH_TRACING": "true",
        }, clear=True):
            with patch("app.langsmith_integration.LangSmithClient", side_effect=TypeError("bad")):
                result = init_langsmith()
                assert result is None
                from app.langsmith_integration import _enabled
                assert _enabled is False


class TestStartTrace:
    def test_start_trace_disabled_returns_none_trace_id(self):
        """start_trace yields {'trace_id': None} when not enabled."""
        with patch("app.langsmith_integration._enabled", False):
            with start_trace("test-trace") as trace:
                assert trace == {"trace_id": None}

    def test_start_trace_enabled_with_client(self):
        """start_trace yields a trace_id when enabled with a client."""
        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.id = "test-run-id-123"
        mock_client.create_run.return_value = mock_run

        with patch("app.langsmith_integration._enabled", True):
            with patch("app.langsmith_integration._client", mock_client):
                with start_trace("test-trace") as trace:
                    assert trace["trace_id"] == "test-run-id-123"
                    mock_client.create_run.assert_called_once_with(
                        name="test-trace", run_type="chain",
                        start_time=unittest.mock.ANY,
                        extra={"trace_name": "test-trace"},
                    )

    def test_start_trace_fallback_to_fake_trace_id(self):
        """start_trace uses a fake trace_id when create_run returns None."""
        mock_client = MagicMock()
        mock_client.create_run.return_value = None

        with patch("app.langsmith_integration._enabled", True):
            with patch("app.langsmith_integration._client", mock_client):
                with start_trace("test-trace") as trace:
                    assert trace["trace_id"] == "ls-test-trace"

    def test_start_trace_calls_update_run_on_exit(self):
        """start_trace calls update_run on context exit."""
        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.id = "run-id"
        mock_client.create_run.return_value = mock_run
        mock_client.update_run = MagicMock()

        with patch("app.langsmith_integration._enabled", True):
            with patch("app.langsmith_integration._client", mock_client):
                with start_trace("test-trace"):
                    pass

        mock_client.update_run.assert_called_once()

    def test_start_trace_handles_shutdown_failure(self):
        """start_trace handles exceptions during update_run gracefully."""
        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.id = "run-id"
        mock_client.create_run.return_value = mock_run
        mock_client.update_run = MagicMock(side_effect=RuntimeError("update failed"))

        with patch("app.langsmith_integration._enabled", True):
            with patch("app.langsmith_integration._client", mock_client):
                with start_trace("test-trace"):
                    pass
        # Should not raise


class TestRecordSpan:
    def test_record_span_disabled_returns_none(self):
        """record_span returns None when not enabled."""
        with patch("app.langsmith_integration._enabled", False):
            result = record_span("test-span", {"key": "value"})
            assert result is None

    def test_record_span_no_client_returns_none(self):
        """record_span returns None when _client is None."""
        with patch("app.langsmith_integration._enabled", True):
            with patch("app.langsmith_integration._client", None):
                result = record_span("test-span", {"key": "value"})
                assert result is None

    def test_record_span_uses_create_feedback(self):
        """record_span uses create_feedback if trace_id is a valid UUID."""
        mock_client = MagicMock()
        mock_client.create_feedback = MagicMock()

        with patch("app.langsmith_integration._enabled", True):
            with patch("app.langsmith_integration._client", mock_client):
                with patch("app.langsmith_integration._last_trace_id", "00000000-0000-0000-0000-000000000001"):
                    result = record_span("test-span", {"key": "value"})
                    assert result is None
                    mock_client.create_feedback.assert_called_once_with(
                        run_id="00000000-0000-0000-0000-000000000001",
                        key="span:test-span",
                        score=None,
                        comment=None,
                        source_info={"metadata": {"key": "value"}},
                    )

    def test_record_span_skips_non_uuid_trace_id(self):
        """record_span skips create_feedback for non-UUID (fallback) trace IDs."""
        mock_client = MagicMock()
        mock_client.create_feedback = MagicMock()

        with patch("app.langsmith_integration._enabled", True):
            with patch("app.langsmith_integration._client", mock_client):
                with patch("app.langsmith_integration._last_trace_id", "ls-fallback-span"):
                    result = record_span("test-span", {"key": "value"})
                    assert result is None
                    mock_client.create_feedback.assert_not_called()

    def test_record_span_create_feedback_failure_caught_gracefully(self):
        """record_span catches exceptions from create_feedback and returns None."""
        mock_client = MagicMock()
        mock_client.create_feedback = MagicMock(side_effect=TypeError("bad"))

        with patch("app.langsmith_integration._enabled", True):
            with patch("app.langsmith_integration._client", mock_client):
                with patch("app.langsmith_integration._last_trace_id", "00000000-0000-0000-0000-000000000001"):
                    result = record_span("test-span", {"key": "value"})
                    assert result is None
                    mock_client.create_feedback.assert_called_once()


class TestGetLangsmithTraceId:
    def test_returns_last_trace_id(self):
        """get_langsmith_trace_id returns the last stored trace_id."""
        with patch("app.langsmith_integration._last_trace_id", "test-trace-123"):
            assert get_langsmith_trace_id() == "test-trace-123"

    def test_returns_none_when_not_set(self):
        """get_langsmith_trace_id returns None when no trace has been started."""
        with patch("app.langsmith_integration._last_trace_id", None):
            assert get_langsmith_trace_id() is None