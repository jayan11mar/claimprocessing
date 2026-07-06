#!/usr/bin/env python3
"""
Simple test to verify knowledge retrieval integration works.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unittest.mock import patch

from app.chains.agent_chain import AgentChain
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent, FAQResponse


def test_knowledge_retrieval_tool_registered():
    """Test that knowledge_retrieval tool is registered."""
    print("\n" + "="*80)
    print("TEST: Knowledge Retrieval Tool Registration")
    print("="*80)
    
    chain = AgentChain(memory=SQLiteMemory())
    tool_names = [tool["name"] for tool in chain.tools]
    
    print(f"Registered tools: {tool_names}")
    assert "knowledge_retrieval" in tool_names, "knowledge_retrieval tool not registered!"
    
    print("\n✓ Knowledge retrieval tool is registered!")


def test_knowledge_retrieval_handler():
    """Test knowledge retrieval handler for policy questions."""
    print("\n" + "="*80)
    print("TEST: Knowledge Retrieval Handler")
    print("="*80)
    
    from unittest.mock import patch
    
    chain = AgentChain(memory=SQLiteMemory())
    
    # Create a mock KNOWLEDGE_RETRIEVAL intent response
    mock_response = FAQResponse(
        intent=FAQIntent.KNOWLEDGE_RETRIEVAL,
        category="coverage",
        confidence=0.9,
        answer_text="Knee replacement exclusions",
        reasoning="User asked about SBI policy",
        metadata={},
    )
    
    # Mock the knowledge_retrieval function
    with patch("app.chains.agent_chain.knowledge_retrieval") as mock_kr:
        mock_kr.return_value = {
            "answer_text": "Joint replacement surgeries including knee replacement may have a waiting period of 2-4 years.",
            "citations": [
                {
                    "source_id": "health_policy_sbi",
                    "source_path": "data/knowledge_base/policies/health_sbihealth_wording.pdf",
                    "score": 0.95,
                }
            ],
            "confidence": 0.92,
            "retrieval_trace": [],
        }
        
        result = chain._handle_knowledge_retrieval(
            mock_response,
            "Are there any exclusions for knee replacement surgery? I have a health insurance policy from SBI.",
            {"tools": []},
            "test-trace-id",
        )
        
        print(f"Query: 'Are there any exclusions for knee replacement surgery? I have a health insurance policy from SBI.'")
        print(f"Answer: {result.answer_text[:100]}...")
        print(f"Confidence: {result.confidence}")
        print(f"Citations: {len(result.metadata.get('citations', []))}")
        
        # Verify knowledge_retrieval was called
        assert mock_kr.called, "knowledge_retrieval was not called!"
        
        # Verify citations are in metadata
        assert "citations" in result.metadata, "Citations not in metadata!"
        assert len(result.metadata["citations"]) > 0, "No citations returned!"
        
        print("\n✓ Knowledge retrieval handler working correctly!")


def test_knowledge_retrieval_routing():
    """Test that KNOWLEDGE_RETRIEVAL intent is routed correctly."""
    print("\n" + "="*80)
    print("TEST: Knowledge Retrieval Routing")
    print("="*80)
    
    chain = AgentChain(memory=SQLiteMemory())
    
    # Create a mock KNOWLEDGE_RETRIEVAL intent response
    mock_response = FAQResponse(
        intent=FAQIntent.KNOWLEDGE_RETRIEVAL,
        category="coverage",
        confidence=0.9,
        answer_text="Test answer",
        reasoning="Test reasoning",
        metadata={},
    )
    
    with patch.object(chain.faq_chain, "invoke", return_value=mock_response):
        with patch.object(chain, "_handle_knowledge_retrieval") as mock_handler:
            mock_handler.return_value = mock_response
            
            result = chain.invoke("test-session", "test question")
            
            # Verify _handle_knowledge_retrieval was called
            assert mock_handler.called, "_handle_knowledge_retrieval was not called!"
            assert result.intent == FAQIntent.KNOWLEDGE_RETRIEVAL
            
            print(f"Intent correctly routed to: {result.intent}")
            print("\n✓ Knowledge retrieval routing working correctly!")


if __name__ == "__main__":
    try:
        test_knowledge_retrieval_tool_registered()
        test_knowledge_retrieval_handler()
        test_knowledge_retrieval_routing()
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
