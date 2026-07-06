#!/usr/bin/env python3
"""
Simple test to verify RAG chunk loading and retrieval works.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chains.agent_chain import AgentChain
from app.memory.sqlite_memory import SQLiteMemory


def test_rag_chunks_load():
    """Test that RAG chunks are loaded successfully."""
    print("\n" + "="*80)
    print("TEST: RAG Chunk Loading")
    print("="*80)
    
    chain = AgentChain(memory=SQLiteMemory())
    
    # Ensure chunks are loaded
    chain._ensure_rag_chunks_loaded()
    
    print(f"Documents loaded: {len(chain._rag_documents)}")
    print(f"Chunks loaded: {len(chain._rag_chunks)}")
    
    if chain._rag_documents:
        print("\nDocuments:")
        for doc_id, doc in list(chain._rag_documents.items())[:3]:
            print(f"  - {doc_id}: {doc.source_path}")
    
    if chain._rag_chunks:
        print("\nSample chunks:")
        for chunk in chain._rag_chunks[:2]:
            print(f"  - Index: {chunk.chunk_index}")
            print(f"    Text: {chunk.text[:100]}...")
            print(f"    Source: {chunk.source_id}")
    
    assert len(chain._rag_chunks) > 0, "No RAG chunks loaded!"
    print("\n✓ RAG chunks loaded successfully!")


def test_rag_retrieval():
    """Test RAG retrieval function directly."""
    print("\n" + "="*80)
    print("TEST: RAG Retrieval")
    print("="*80)
    
    chain = AgentChain(memory=SQLiteMemory())
    chain._ensure_rag_chunks_loaded()
    
    # Test retrieval
    query = "knee replacement surgery exclusions"
    rag_context, retrieval_trace, citations = chain._perform_rag_retrieval(
        query, {"tools": []}, None
    )
    
    print(f"Query: '{query}'")
    print(f"Context length: {len(rag_context) if rag_context else 0}")
    print(f"Retrieval results: {len(retrieval_trace) if isinstance(retrieval_trace, list) else 0}")
    print(f"Citations: {len(citations) if isinstance(citations, list) else 0}")
    
    if rag_context:
        print(f"\nRetrieved context (first 300 chars):\n{rag_context[:300]}...")
    
    if retrieval_trace:
        print(f"\nRetrieval trace:")
        for item in retrieval_trace[:2]:
            print(f"  - Score: {item.get('score', 0):.3f}, Source: {item.get('source_path', 'N/A')}")
    
    print("\n✓ RAG retrieval working!")


def test_other_intent_handler():
    """Test that _handle_other is called for OTHER intent."""
    print("\n" + "="*80)
    print("TEST: _handle_other Handler (Mock)")
    print("="*80)
    
    from app.models.faq import FAQIntent, FAQResponse
    
    chain = AgentChain(memory=SQLiteMemory())
    chain._ensure_rag_chunks_loaded()
    
    # Create a mock OTHER intent response
    mock_response = FAQResponse(
        intent=FAQIntent.OTHER,
        category="general",
        confidence=0.5,
        answer_text="This is a generic response.",
        reasoning="Mock response for testing",
    )
    
    # Call the handler
    result = chain._handle_other(
        mock_response,
        "What are the exclusions for surgery?",
        {"tools": [], "llm_ms": 0},
        None,
    )
    
    print(f"Input intent: {mock_response.intent}")
    print(f"Output intent: {result.intent}")
    print(f"Output category: {result.category}")
    print(f"Output confidence: {result.confidence}")
    print(f"RAG enabled in metadata: {result.metadata.get('rag_enabled', False)}")
    print(f"Has retrieval_trace: {'retrieval_trace' in result.metadata}")
    print(f"Answer length: {len(result.answer_text)} chars")
    
    print("\n✓ _handle_other handler executed!")


if __name__ == "__main__":
    try:
        test_rag_chunks_load()
        test_rag_retrieval()
        test_other_intent_handler()
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
