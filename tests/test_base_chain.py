"""Tests for app/chains/base_chain.py to achieve >85% coverage."""

from typing import List, Dict
from unittest.mock import patch, MagicMock

import pytest

from app.chains.base_chain import (
    get_chat_model,
    format_examples_block,
    format_examples_from_objects,
    build_faq_prompt,
    build_faq_prompt_with_history,
)
from app.prompts.examples_store import Example


class TestGetChatModel:
    def test_returns_none_when_no_api_key(self):
        """get_chat_model returns None when OPENAI_API_KEY is missing or placeholder."""
        with patch("app.chains.base_chain.get_settings") as mock_settings:
            settings = MagicMock()
            settings.OPENAI_API_KEY = ""
            settings.OPENAI_MODEL_NAME = "gpt-4o-mini"
            settings.OPENAI_MODEL_TEMPERATURE = 0.8
            settings.OPENAI_REQUEST_TIMEOUT = 30.0
            mock_settings.return_value = settings

            model = get_chat_model()
            assert model is None

    def test_returns_none_when_api_key_is_placeholder(self):
        """get_chat_model returns None when OPENAI_API_KEY starts with 'your-'."""
        with patch("app.chains.base_chain.get_settings") as mock_settings:
            settings = MagicMock()
            settings.OPENAI_API_KEY = "your-api-key-here"
            settings.OPENAI_MODEL_NAME = "gpt-4o-mini"
            settings.OPENAI_MODEL_TEMPERATURE = 0.8
            settings.OPENAI_REQUEST_TIMEOUT = 30.0
            mock_settings.return_value = settings

            model = get_chat_model()
            assert model is None

    def test_returns_chat_openai_when_api_key_present(self):
        """get_chat_model returns a ChatOpenAI instance when a valid API key is set."""
        with patch("app.chains.base_chain.get_settings") as mock_settings:
            settings = MagicMock()
            settings.OPENAI_API_KEY = "sk-real-key"
            settings.OPENAI_MODEL_NAME = "gpt-4o-mini"
            settings.OPENAI_MODEL_TEMPERATURE = 0.8
            settings.OPENAI_REQUEST_TIMEOUT = 30.0
            mock_settings.return_value = settings

            with patch("app.chains.base_chain.ChatOpenAI") as mock_chat:
                mock_instance = MagicMock()
                mock_chat.return_value = mock_instance

                model = get_chat_model()
                assert model is mock_instance
                mock_chat.assert_called_once_with(
                    model="gpt-4o-mini",
                    temperature=0.8,
                    openai_api_key="sk-real-key",
                    timeout=30.0,
                    max_retries=2,
                )


class TestFormatExamplesBlock:
    def test_formats_single_example(self):
        examples = [{"user": "Hello", "assistant": "Hi there!"}]
        result = format_examples_block(examples)
        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result

    def test_formats_multiple_examples(self):
        examples = [
            {"user": "Hello", "assistant": "Hi there!"},
            {"user": "How are you?", "assistant": "I'm fine, thanks!"},
        ]
        result = format_examples_block(examples)
        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result
        assert "User: How are you?" in result
        assert "Assistant: I'm fine, thanks!" in result
        # Multiple examples should be separated by double newlines
        assert "\n\n" in result

    def test_handles_empty_list(self):
        result = format_examples_block([])
        assert result == ""


class TestFormatExamplesFromObjects:
    def test_converts_example_objects_to_dicts(self):
        examples = [
            Example(user="Hello", assistant="Hi there!", intent="OTHER", category="general"),
            Example(user="How are you?", assistant="I'm fine!", intent="OTHER", category="general"),
        ]
        result = format_examples_from_objects(examples)
        assert len(result) == 2
        assert result[0] == {"user": "Hello", "assistant": "Hi there!"}
        assert result[1] == {"user": "How are you?", "assistant": "I'm fine!"}

    def test_handles_empty_list(self):
        result = format_examples_from_objects([])
        assert result == []


class TestBuildFaqPrompt:
    def test_returns_chat_prompt_template(self):
        examples = [{"user": "Hello", "assistant": "Hi"}]
        prompt = build_faq_prompt(examples)
        assert prompt is not None
        # Should have system + human messages
        messages = prompt.messages
        assert len(messages) == 2

    def test_includes_examples_in_prompt(self):
        examples = [{"user": "Test user", "assistant": "Test assistant"}]
        prompt = build_faq_prompt(examples)
        # Format the prompt to verify examples are included
        formatted = prompt.format_prompt(
            json_instruction="Return JSON",
            example_block="User: Test user\nAssistant: Test assistant",
            user_message="Hello",
        )
        messages = formatted.to_messages()
        combined = " ".join(m.content for m in messages)
        assert "Test user" in combined
        assert "Test assistant" in combined


class TestBuildFaqPromptWithHistory:
    def test_returns_chat_prompt_template(self):
        prompt = build_faq_prompt_with_history()
        assert prompt is not None
        messages = prompt.messages
        assert len(messages) == 2

    def test_includes_history_placeholder(self):
        prompt = build_faq_prompt_with_history()
        formatted = prompt.format_prompt(
            json_instruction="Return JSON",
            example_block="User: Hi\nAssistant: Hello",
            history="User: previous message\nAssistant: previous response",
            user_message="Hello",
        )
        messages = formatted.to_messages()
        combined = " ".join(m.content for m in messages)
        assert "Conversation History" in combined
        assert "previous message" in combined