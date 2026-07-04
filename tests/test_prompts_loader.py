"""Tests for app/prompts/loader.py to achieve >85% coverage."""

from unittest.mock import patch, MagicMock, mock_open
import json
import pytest

from app.prompts.loader import (
    load_templates,
    get_system_template,
    get_guardrail_response,
    get_few_shot_examples,
    get_json_format_instruction,
    _TEMPLATES_CACHE,
)


SAMPLE_TEMPLATES = {
    "system": {
        "main_faq_assistant": "You are a helpful FAQ assistant.",
        "json_format_instruction": "Return your response as JSON.",
    },
    "guardrails": {
        "blocked_keywords": "I cannot process that request.",
        "pii_detected": "Please do not share personal information.",
    },
    "few_shot_examples": {
        "example1": {"user": "What is my policy status?", "assistant": "Your policy is active."},
        "example2": {"user": "File a claim", "assistant": "Please provide your policy number."},
    },
}


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the template cache before each test."""
    _TEMPLATES_CACHE.clear()
    yield


class TestLoadTemplates:
    def test_loads_templates_from_file(self):
        """load_templates reads and caches templates from templates.json."""
        # Mock the actual file path that get_system_template uses internally
        with patch("app.prompts.loader.Path") as mock_path_cls:
            mock_path_instance = MagicMock()
            mock_path_cls.return_value = mock_path_instance
            mock_path_instance.exists.return_value = True
            mock_path_instance.__truediv__.return_value = mock_path_instance

            with patch("builtins.open", mock_open(read_data=json.dumps(SAMPLE_TEMPLATES))):
                templates = load_templates()
                assert templates == SAMPLE_TEMPLATES
                assert _TEMPLATES_CACHE == SAMPLE_TEMPLATES

    def test_returns_cached_templates(self):
        """load_templates returns cached templates without reading the file."""
        _TEMPLATES_CACHE["cached"] = True

        with patch("app.prompts.loader.Path") as mock_path_cls:
            templates = load_templates()
            assert templates == {"cached": True}
            # Path should NOT be called if cache is populated
            mock_path_cls.assert_not_called()

    def test_raises_file_not_found(self):
        """load_templates raises FileNotFoundError when templates.json doesn't exist."""
        with patch("app.prompts.loader.Path") as mock_path_cls:
            mock_path_instance = MagicMock()
            mock_path_cls.return_value = mock_path_instance
            mock_path_instance.exists.return_value = False
            mock_path_instance.__truediv__.return_value = mock_path_instance

            with pytest.raises(FileNotFoundError, match="Templates file not found"):
                load_templates()


class TestGetSystemTemplate:
    def test_returns_system_template(self):
        """get_system_template returns the correct system template."""
        _TEMPLATES_CACHE.update(SAMPLE_TEMPLATES)
        result = get_system_template("main_faq_assistant")
        assert result == "You are a helpful FAQ assistant."

    def test_returns_empty_string_for_missing_template(self):
        """get_system_template returns empty string for unknown template name."""
        _TEMPLATES_CACHE.update(SAMPLE_TEMPLATES)
        result = get_system_template("nonexistent_template")
        assert result == ""

    def test_uses_default_template_name(self):
        """get_system_template uses 'main_faq_assistant' as default."""
        _TEMPLATES_CACHE.update(SAMPLE_TEMPLATES)
        result = get_system_template()
        assert result == "You are a helpful FAQ assistant."


class TestGetGuardrailResponse:
    def test_returns_guardrail_response(self):
        """get_guardrail_response returns the correct guardrail message."""
        _TEMPLATES_CACHE.update(SAMPLE_TEMPLATES)
        result = get_guardrail_response("blocked_keywords")
        assert result == "I cannot process that request."

    def test_returns_default_for_missing_rule(self):
        """get_guardrail_response returns default message for unknown rule."""
        _TEMPLATES_CACHE.update(SAMPLE_TEMPLATES)
        result = get_guardrail_response("unknown_rule")
        assert result == "I cannot assist with that request."


class TestGetFewShotExamples:
    def test_returns_few_shot_examples(self):
        """get_few_shot_examples returns the list of example dicts."""
        _TEMPLATES_CACHE.update(SAMPLE_TEMPLATES)
        examples = get_few_shot_examples()
        assert len(examples) == 2
        assert examples[0]["user"] == "What is my policy status?"
        assert examples[1]["user"] == "File a claim"

    def test_returns_empty_list_when_no_examples(self):
        """get_few_shot_examples returns empty list when no examples configured."""
        _TEMPLATES_CACHE.update({"system": {}, "guardrails": {}})
        examples = get_few_shot_examples()
        assert examples == []


class TestGetJsonFormatInstruction:
    def test_returns_json_format_instruction(self):
        """get_json_format_instruction returns the JSON format instruction."""
        _TEMPLATES_CACHE.update(SAMPLE_TEMPLATES)
        result = get_json_format_instruction()
        assert result == "Return your response as JSON."

    def test_returns_empty_string_when_not_configured(self):
        """get_json_format_instruction returns empty string when not in templates."""
        _TEMPLATES_CACHE.update({"system": {}})
        result = get_json_format_instruction()
        assert result == ""