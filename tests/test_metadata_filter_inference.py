"""Tests for metadata filter inference in agent_chain.py.

Validates that health-related policy queries are correctly identified and
not incorrectly filtered as motor insurance queries.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.chains.agent_chain import AgentChain
from app.models.faq import FAQIntent, FAQResponse


class TestMetadataFilterInference:
    """Test suite for _infer_insurance_type method."""

    @pytest.fixture
    def agent_chain(self):
        """Create an AgentChain instance with mocked memory."""
        with patch("app.chains.agent_chain.SQLiteMemory"):
            return AgentChain()

    def test_day_care_procedures_detected_as_health(self, agent_chain):
        """Query 1: 'What is covered under day care procedures in this policy?'
        Expected: insurance_type = health"""
        result = agent_chain._infer_insurance_type(
            "What is covered under day care procedures in this policy"
        )
        assert result == "health", "Day care procedures should be detected as health insurance"

    def test_own_damage_detected_as_motor(self, agent_chain):
        """Query 2: 'What is covered under own damage in this policy?'
        Expected: insurance_type = motor"""
        result = agent_chain._infer_insurance_type(
            "What is covered under own damage in this policy"
        )
        assert result == "motor", "Own damage should be detected as motor insurance"

    def test_exclusions_without_context_returns_none(self, agent_chain):
        """Query 3: 'What are the exclusions in this policy?'
        Expected: insurance_type = None unless context is available"""
        result = agent_chain._infer_insurance_type(
            "What are the exclusions in this policy"
        )
        assert result is None, "Generic exclusions query without context should return None"

    def test_hospitalization_detected_as_health(self, agent_chain):
        """Query 4: 'What is covered for hospitalization?'
        Expected: insurance_type = health"""
        result = agent_chain._infer_insurance_type(
            "What is covered for hospitalization"
        )
        assert result == "health", "Hospitalization should be detected as health insurance"

    def test_vehicle_accident_damage_detected_as_motor(self, agent_chain):
        """Query 5: 'What is covered for vehicle accident damage?'
        Expected: insurance_type = motor"""
        result = agent_chain._infer_insurance_type(
            "What is covered for vehicle accident damage"
        )
        assert result == "motor", "Vehicle accident damage should be detected as motor insurance"

    def test_health_keywords_comprehensive(self, agent_chain):
        """Test various health-related keywords."""
        health_queries = [
            "What is the medical expense coverage?",
            "Is surgery covered under this policy?",
            "What is the room rent limit?",
            "Is ICU charges covered?",
            "Is ambulance service available?",
            "Is cashless claim available at network hospitals?",
            "What is the pre-existing disease waiting period?",
            "Is OPD consultation covered?",
            "What is the inpatient treatment coverage?",
            "Is domiciliary hospitalization covered?",
        ]
        for query in health_queries:
            result = agent_chain._infer_insurance_type(query)
            assert result == "health", f"Failed for query: {query}"

    def test_motor_keywords_comprehensive(self, agent_chain):
        """Test various motor-related keywords."""
        motor_queries = [
            "What is the IDV of my car?",
            "Is engine damage covered?",
            "What is the tyre coverage?",
            "Is bumper damage covered?",
            "Is windshield damage covered?",
            "What is the third party liability?",
            "Is theft covered under comprehensive?",
            "What is the NCB discount?",
            "Is garage repair available?",
            "What is the fuel type coverage?",
        ]
        for query in motor_queries:
            result = agent_chain._infer_insurance_type(query)
            assert result == "motor", f"Failed for query: {query}"

    def test_conflicting_keywords_returns_none(self, agent_chain):
        """Test that conflicting health and motor keywords return None."""
        # These queries contain both health and motor keywords
        conflicting_queries = [
            "What is covered for motor vehicle accident with hospitalization?",
            "Is bike accident injury treatment covered?",
            "What is the coverage for vehicle accident with medical expenses?",
        ]
        for query in conflicting_queries:
            result = agent_chain._infer_insurance_type(query)
            assert result is None, f"Conflicting keywords should return None for: {query}"

    def test_unknown_queries_return_none(self, agent_chain):
        """Test that queries without clear insurance type return None."""
        unknown_queries = [
            "What is the policy term?",
            "How do I renew my policy?",
            "Tell me about the policy features",
            "What documents are required?",
        ]
        for query in unknown_queries:
            result = agent_chain._infer_insurance_type(query)
            assert result is None, f"Unknown query should return None for: {query}"

    def test_case_insensitive_matching(self, agent_chain):
        """Test that keyword matching is case-insensitive."""
        queries = [
            ("What is DAY CARE coverage?", "health"),
            ("Is HOSPITALIZATION covered?", "health"),
            ("What is the IDV of my CAR?", "motor"),
            ("Is ENGINE damage covered?", "motor"),
        ]
        for query, expected in queries:
            result = agent_chain._infer_insurance_type(query)
            assert result == expected, f"Case insensitive matching failed for: {query}"

    def test_metadata_filter_creation_with_logging(self, agent_chain):
        """Test that metadata filter is created correctly with logging."""
        mock_intent = FAQResponse(
            intent=FAQIntent.KNOWLEDGE_RETRIEVAL,
            category="policy_coverage",
            confidence=0.9,
            answer_text="",
            reasoning="",
            metadata={},
        )
        
        mock_timings = {"llm_ms": 100, "tools": []}
        mock_trace_id = "test-trace-123"
        
        with patch("app.chains.agent_chain.knowledge_retrieval") as mock_knowledge_retrieval:
            mock_knowledge_retrieval.return_value = {
                "answer_text": "Test answer",
                "citations": [],
                "confidence": 0.8,
            }
            
            with patch("app.chains.agent_chain.logger") as mock_logger:
                result = agent_chain._handle_knowledge_retrieval(
                    mock_intent,
                    "What is covered under day care procedures?",
                    mock_timings,
                    mock_trace_id,
                )
                
                # Verify knowledge_retrieval was called with correct metadata_filter
                call_args = mock_knowledge_retrieval.call_args
                assert call_args is not None
                metadata_filter = call_args.kwargs.get("metadata_filter")
                assert metadata_filter == {"insurance_type": "health"}
                
                # Verify logging was called
                assert mock_logger.info.called

    def test_metadata_filter_none_for_ambiguous_query(self, agent_chain):
        """Test that metadata_filter is None for ambiguous queries."""
        mock_intent = FAQResponse(
            intent=FAQIntent.KNOWLEDGE_RETRIEVAL,
            category="policy_coverage",
            confidence=0.9,
            answer_text="",
            reasoning="",
            metadata={},
        )
        
        mock_timings = {"llm_ms": 100, "tools": []}
        mock_trace_id = "test-trace-123"
        
        with patch("app.chains.agent_chain.knowledge_retrieval") as mock_knowledge_retrieval:
            mock_knowledge_retrieval.return_value = {
                "answer_text": "Test answer",
                "citations": [],
                "confidence": 0.8,
            }
            
            result = agent_chain._handle_knowledge_retrieval(
                mock_intent,
                "What are the exclusions in this policy?",
                mock_timings,
                mock_trace_id,
            )
            
            # Verify knowledge_retrieval was called with None metadata_filter
            call_args = mock_knowledge_retrieval.call_args
            assert call_args is not None
            metadata_filter = call_args.kwargs.get("metadata_filter")
            assert metadata_filter is None

    def test_logging_contains_required_fields(self, agent_chain):
        """Test that defensive logging contains all required fields."""
        with patch("app.chains.agent_chain.logger") as mock_logger:
            agent_chain._infer_insurance_type("What is covered under day care procedures?")
            
            # Check that logger.info was called
            assert mock_logger.info.called
            
            # Get all calls and find the one with 'extra' parameter
            found_extra = False
            for call in mock_logger.info.call_args_list:
                extra = call.kwargs.get("extra", {})
                if "query" in extra:
                    found_extra = True
                    # Verify required fields are present
                    assert "health_keywords_found" in extra
                    assert "motor_keywords_found" in extra
                    assert extra["query"] == "What is covered under day care procedures?"
                    break
            
            assert found_extra, "No logger.info call found with 'extra' parameter containing 'query'"
