#!/usr/bin/env python3
"""
Quick test to verify RAG integration for FAQIntent.OTHER questions.
"""
import sys
from pathlib import Path

# Add the project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chains.agent_chain import AgentChain
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent


def test_rag_integration():
    """Test that RAG is invoked for general policy questions."""
    
    memory = SQLiteMemory()
    chain = AgentChain(memory=memory)
    
    session_id = "test-rag-session"
    
    # Test 1: General knowledge question that should trigger RAG
    print("\n" + "="*80)
    print("TEST 1: General Policy Question (should use RAG)")
    print("="*80)
    question = "Are there any exclusions for knee replacement surgery?"
    response = chain.invoke(session_id, question)
    
    print(f"\nQuestion: {question}")
    print(f"Intent: {response.intent}")
    print(f"Category: {response.category}")
    print(f"Confidence: {response.confidence}")
    print(f"Answer: {response.answer_text[:200]}...")
    print(f"\nMetadata Keys: {list(response.metadata.keys())}")
    
    if response.metadata:
        rag_enabled = response.metadata.get("rag_enabled", False)
        print(f"RAG Enabled: {rag_enabled}")
        
        if "retrieval_trace" in response.metadata:
            trace = response.metadata["retrieval_trace"]
            print(f"Retrieval Results: {len(trace) if isinstance(trace, list) else 'N/A'}")
            if isinstance(trace, list) and trace:
                for item in trace[:2]:
                    print(f"  - Source: {item.get('source_path', 'N/A')}, Score: {item.get('score', 0):.3f}")
        
        if "citations" in response.metadata:
            citations = response.metadata["citations"]
            print(f"Citations: {len(citations) if isinstance(citations, list) else 'N/A'}")
    
    # Test 2: Another general question
    print("\n" + "="*80)
    print("TEST 2: Another General Knowledge Question")
    print("="*80)
    question2 = "What is covered under the health insurance policy?"
    response2 = chain.invoke(session_id, question2)
    
    print(f"\nQuestion: {question2}")
    print(f"Intent: {response2.intent}")
    print(f"Category: {response2.category}")
    print(f"Answer: {response2.answer_text[:200]}...")
    
    if response2.metadata:
        rag_enabled = response2.metadata.get("rag_enabled", False)
        print(f"RAG Enabled: {rag_enabled}")
    
    # Test 3: Claim-specific question (should NOT use RAG)
    print("\n" + "="*80)
    print("TEST 3: Claim Registration (should NOT use RAG)")
    print("="*80)
    question3 = "I want to register a claim with policy P123456 for $5000"
    response3 = chain.invoke(session_id, question3)
    
    print(f"\nQuestion: {question3}")
    print(f"Intent: {response3.intent}")
    print(f"Category: {response3.category}")
    print(f"Answer: {response3.answer_text[:200]}...")
    
    print("\n" + "="*80)
    print("RAG Integration Tests Complete!")
    print("="*80)


if __name__ == "__main__":
    test_rag_integration()
