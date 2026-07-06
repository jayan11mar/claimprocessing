"""Tests for knowledge retrieval integration with agent chain."""

import pytest
from unittest.mock import patch, MagicMock

from app.chains.agent_chain import AgentChain
from app.models.faq import FAQIntent, FAQResponse


def test_agent_chain_has_knowledge_retrieval_tool():
    """Test that knowledge_retrieval tool is registered in agent chain."""
    agent = AgentChain()
    tool_names = [tool["name"] for tool in agent.tools]
    assert "knowledge_retrieval" in tool_names


def test_agent_chain_handles_knowledge_retrieval_query():
    """Test that the agent chain properly handles knowledge retrieval queries."""
    agent = AgentChain()
    
    # Create a mock FAQResponse with KNOWLEDGE_RETRIEVAL intent
    mock_response = FAQResponse(
        intent=FAQIntent.KNOWLEDGE_RETRIEVAL,
        category="coverage",
        confidence=0.9,
        answer_text="Knee replacement surgery has a waiting period of 2-4 years.",
        reasoning="User asked about SBI policy exclusions",
        metadata={},
    )
    
    # Mock the knowledge_retrieval function
    with patch("app.chains.agent_chain.knowledge_retrieval") as mock_kr:
        mock_kr.return_value = {
            "answer_text": "Joint replacement surgeries including knee replacement may have a waiting period of 2-4 years depending on the policy. Pre-existing degenerative conditions may be excluded.",
            "citations": [
                {
                    "chunk_id": "health_policy_sbi_0",
                    "text": "Knee replacement surgery exclusion clause...",
                    "source_id": "health_policy_sbi",
                    "source_path": "data/knowledge_base/policies/health_sbihealth_wording.pdf",
                    "score": 0.95,
                }
            ],
            "confidence": 0.92,
            "retrieval_trace": [
                {
                    "tool": "knowledge_retrieval",
                    "query": "Are there any exclusions for knee replacement surgery? I have a health insurance policy from SBI.",
                    "top_k": 3,
                    "metadata_filter": {"insurance_type": "health"},
                    "result_count": 1,
                }
            ],
        }
        
        result = agent._handle_knowledge_retrieval(
            mock_response,
            "Are there any exclusions for knee replacement surgery? I have a health insurance policy from SBI.",
            {"llm_ms": 0, "tools": []},
            "test-trace-id",
        )
        
        # Verify knowledge_retrieval was called with correct parameters
        mock_kr.assert_called_once()
        call_args = mock_kr.call_args
        assert call_args[1]["query"] == "Are there any exclusions for knee replacement surgery? I have a health insurance policy from SBI."
        assert call_args[1]["top_k"] == 3
        assert call_args[1]["metadata_filter"] == {"insurance_type": "health"}
        
        # Verify the result contains citations
        assert "citations" in result.metadata
        assert len(result.metadata["citations"]) == 1
        assert result.metadata["citations"][0]["source_id"] == "health_policy_sbi"
        
        # Verify the answer text is from the knowledge retrieval
        assert "waiting period of 2-4 years" in result.answer_text


def test_agent_chain_handles_knowledge_retrieval_motor_query():
    """Test that the agent chain properly filters by insurance type for motor queries."""
    agent = AgentChain()
    
    # Create a mock FAQResponse with KNOWLEDGE_RETRIEVAL intent
    mock_response = FAQResponse(
        intent=FAQIntent.KNOWLEDGE_RETRIEVAL,
        category="coverage",
        confidence=0.9,
        answer_text="Motor insurance claim settlement time",
        reasoning="User asked about motor claim settlement",
        metadata={},
    )
    
    # Mock the knowledge_retrieval function
    with patch("app.chains.agent_chain.knowledge_retrieval") as mock_kr:
        mock_kr.return_value = {
            "answer_text": "Motor insurance claims are typically settled within 5-7 business days.",
            "citations": [],
            "confidence": 0.88,
            "retrieval_trace": [],
        }
        
        result = agent._handle_knowledge_retrieval(
            mock_response,
            "How long does it take to settle a motor insurance claim?",
            {"llm_ms": 0, "tools": []},
            "test-trace-id",
        )
        
        # Verify knowledge_retrieval was called with motor insurance filter
        call_args = mock_kr.call_args
        assert call_args[1]["metadata_filter"] == {"insurance_type": "motor"}


def test_agent_chain_handles_knowledge_retrieval_general_query():
    """Test that the agent chain handles general queries without insurance type filter."""
    agent = AgentChain()
    
    # Create a mock FAQResponse with KNOWLEDGE_RETRIEVAL intent
    mock_response = FAQResponse(
        intent=FAQIntent.KNOWLEDGE_RETRIEVAL,
        category="general",
        confidence=0.85,
        answer_text="General insurance information",
        reasoning="User asked a general question",
        metadata={},
    )
    
    # Mock the knowledge_retrieval function
    with patch("app.chains.agent_chain.knowledge_retrieval") as mock_kr:
        mock_kr.return_value = {
            "answer_text": "Here is some general information about insurance claims.",
            "citations": [],
            "confidence": 0.85,
            "retrieval_trace": [],
        }
        
        result = agent._handle_knowledge_retrieval(
            mock_response,
            "What is the average processing time for claims?",
            {"llm_ms": 0, "tools": []},
            "test-trace-id",
        )
        
        # Verify knowledge_retrieval was called without insurance type filter
        call_args = mock_kr.call_args
        assert call_args[1]["metadata_filter"] is None


def test_knowledge_retrieval_routing_in_invoke():
    """Test that KNOWLEDGE_RETRIEVAL intent is properly routed in invoke method."""
    agent = AgentChain()
    
    # Create a mock FAQResponse with KNOWLEDGE_RETRIEVAL intent
    mock_response = FAQResponse(
        intent=FAQIntent.KNOWLEDGE_RETRIEVAL,
        category="coverage",
        confidence=0.9,
        answer_text="Test answer",
        reasoning="Test reasoning",
        metadata={},
    )
    
    with patch.object(agent.faq_chain, "invoke", return_value=mock_response):
        with patch.object(agent, "_handle_knowledge_retrieval") as mock_handler:
            mock_handler.return_value = mock_response
            
            result = agent.invoke("test-session", "test question")
            
            # Verify _handle_knowledge_retrieval was called
            mock_handler.assert_called_once()
            assert result.intent == FAQIntent.KNOWLEDGE_RETRIEVAL