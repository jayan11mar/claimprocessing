"""Tests for app/logging/json_logger.py to achieve >85% coverage."""

import json
import logging
from unittest.mock import patch, MagicMock

import pytest

from app.logging.json_logger import JsonFormatter, get_logger


class TestJsonFormatter:
    def test_formats_basic_log_record(self):
        """JsonFormatter produces valid JSON with expected fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        payload = json.loads(result)
        assert payload["level"] == "INFO"
        assert payload["message"] == "Test message"
        assert "timestamp" in payload
        assert "timestamp_ms" in payload

    def test_formats_with_dict_args(self):
        """JsonFormatter includes dict args as top-level fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="/test/path.py",
            lineno=42,
            msg="test_event",
            args={"session_id": "s1", "latency_ms": 150},
            exc_info=None,
        )
        result = formatter.format(record)
        payload = json.loads(result)
        assert payload["message"] == "test_event"
        assert payload["session_id"] == "s1"
        assert payload["latency_ms"] == 150

    def test_formats_with_non_dict_args(self):
        """JsonFormatter includes non-dict args as 'args' field."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Error occurred: %s",
            args=("something broke",),
            exc_info=None,
        )
        result = formatter.format(record)
        payload = json.loads(result)
        assert "args" in payload
        # JSON serialization converts tuples to lists
        assert payload["args"] == ["something broke"]

    def test_formats_with_type_error_on_args(self):
        """JsonFormatter handles TypeError when processing args."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="/test/path.py",
            lineno=42,
            msg="test",
            args=None,
            exc_info=None,
        )
        # Manually set args to a non-iterable that will cause TypeError
        # when the formatter tries to iterate over it
        record.args = object()
        result = formatter.format(record)
        payload = json.loads(result)
        assert "args" in payload

    def test_formats_with_extra_fields(self):
        """JsonFormatter includes extra attributes from the record."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="test",
            args=None,
            exc_info=None,
        )
        # Add extra attributes
        record.custom_field = "custom_value"
        record.another_field = 123

        result = formatter.format(record)
        payload = json.loads(result)
        assert payload["custom_field"] == "custom_value"
        assert payload["another_field"] == 123

    def test_excludes_standard_log_record_attributes(self):
        """JsonFormatter excludes standard LogRecord attributes from extra fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="test",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        payload = json.loads(result)
        # Standard attributes should not appear in the output (except the ones we explicitly include)
        assert "name" not in payload
        assert "pathname" not in payload
        assert "lineno" not in payload
        assert "exc_info" not in payload

    def test_handles_non_serializable_values(self):
        """JsonFormatter uses default=str for non-serializable values."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="test",
            args=None,
            exc_info=None,
        )
        # Manually set args to a dict with a non-serializable value
        class NonSerializable:
            def __str__(self):
                return "non_serializable_value"
        record.args = {"obj": NonSerializable()}
        result = formatter.format(record)
        payload = json.loads(result)
        assert payload["obj"] == "non_serializable_value"

    def test_args_keys_do_not_override_payload(self):
        """JsonFormatter does not let args override reserved payload fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="test",
            args={"message": "should not override", "timestamp": "should not override"},
            exc_info=None,
        )
        result = formatter.format(record)
        payload = json.loads(result)
        # The original message and timestamp should remain
        assert payload["message"] == "test"
        assert "timestamp" in payload

    def test_extra_keys_do_not_override_payload(self):
        """JsonFormatter does not let extra fields override reserved payload fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="test",
            args=None,
            exc_info=None,
        )
        # Add extra with key that conflicts with payload
        record.message = "extra message"

        result = formatter.format(record)
        payload = json.loads(result)
        # The original message should remain
        assert payload["message"] == "test"

    def test_formats_debug_level(self):
        """JsonFormatter correctly formats DEBUG level."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.DEBUG,
            pathname="/test/path.py",
            lineno=42,
            msg="Debug message",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        payload = json.loads(result)
        assert payload["level"] == "DEBUG"

    def test_formats_critical_level(self):
        """JsonFormatter correctly formats CRITICAL level."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.CRITICAL,
            pathname="/test/path.py",
            lineno=42,
            msg="Critical message",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        payload = json.loads(result)
        assert payload["level"] == "CRITICAL"


class TestGetLogger:
    def test_returns_logger_with_json_formatter(self):
        """get_logger returns a logger with JSON formatter."""
        logger = get_logger("test_json_logger")
        assert logger.name == "test_json_logger"
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0].formatter, JsonFormatter)

    def test_does_not_add_duplicate_handlers(self):
        """get_logger does not add duplicate handlers if called twice."""
        logger = get_logger("test_duplicate_logger")
        initial_count = len(logger.handlers)
        logger2 = get_logger("test_duplicate_logger")
        assert len(logger2.handlers) == initial_count

    def test_logger_propagate_is_false(self):
        """get_logger sets propagate to False."""
        logger = get_logger("test_no_propagate")
        assert logger.propagate is False

    def test_logger_level_is_info(self):
        """get_logger sets level to INFO."""
        logger = get_logger("test_level_info")
        assert logger.level == logging.INFO