"""
Generate eval set and golden set from knowledge base documents.

This script:
1. Loads all documents from data/knowledge_base
2. Extracts relevant Q&A pairs based on document content
3. Generates an eval set (test queries)
4. Generates a golden set (expected answers with citations)
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.loaders import load_documents_from_manifest
from app.rag.chunkers import chunk_document, ChunkConfig


# Knowledge base directory
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
EVAL_DIR = Path(__file__).parent.parent / "eval"
GOLDEN_DATASET_DIR = Path(__file__).parent.parent / "data" / "golden_dataset"


def extract_qa_from_documents(documents: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract Q&A pairs from loaded documents based on content analysis.
    
    Returns list of dictionaries with:
    - query: question text
    - expected_answer: expected answer based on document content
    - expected_chunks: keywords that should be in retrieved chunks
    - source_doc: source document ID
    - difficulty: easy/medium/hard
    """
    qa_pairs = []
    
    # Define question templates for different document types
    question_templates = {
        "policy_wording": [
            "What is covered under {topic} in this policy?",
            "Are there any exclusions for {topic}?",
            "What is the waiting period for {topic}?",
            "What documents are required for {topic} claims?",
            "Is {topic} covered under this health insurance policy?",
        ],
        "regulation": [
            "What does the regulation say about {topic}?",
            "What are the IRDAI guidelines for {topic}?",
            "Explain the requirements for {topic} under the regulations.",
        ],
        "exclusion_summary": [
            "What are the common exclusions for {topic}?",
            "Is {topic} typically excluded from health insurance coverage?",
            "What surgical procedures are excluded for {topic}?",
        ],
        "network": [
            "What are the network hospital benefits for {topic}?",
            "How does the network agreement affect {topic}?",
        ],
        "memo": [
            "What was the decision in cases involving {topic}?",
            "What factors are considered for {topic} claims?",
        ],
    }
    
    # Define topics to query based on document content
    topics = {
        "health": [
            "knee replacement surgery",
            "pre-existing diseases",
            "maternity benefits",
            "day care procedures",
            "pre-hospitalization",
            "post-hospitalization",
            "ICU charges",
            "AYUSH treatments",
            "newborn coverage",
            "senior citizen",
        ],
        "motor": [
            "total loss",
            "third party liability",
            "own damage",
            "deductible",
            "no claim bonus",
            "flood damage",
        ],
        "general": [
            "portability",
            "claim settlement",
            "network hospitals",
            "cashless hospitalization",
            "reimbursement claims",
        ],
    }
    
    # Process each document
    for doc in documents:
        doc_text = doc.text.lower()
        insurance_type = getattr(doc, 'insurance_type', 'general')
        doc_type = getattr(doc, 'doc_type', '')
        source_id = getattr(doc, 'source_id', 'unknown')
        
        # Get relevant topics for this insurance type
        relevant_topics = topics.get(insurance_type, topics['general'])
        
        # Get question templates for this doc type
        templates = question_templates.get(doc_type, question_templates['policy_wording'])
        
        # Generate Q&A pairs
        for topic in relevant_topics:
            # Check if topic is mentioned in document
            if topic.replace('_', ' ') in doc_text or topic in doc_text:
                # Select appropriate template
                for template in templates[:2]:  # Use first 2 templates per topic
                    query = template.format(topic=topic.replace('_', ' '))
                    
                    # Extract relevant text snippet from document
                    relevant_snippet = extract_relevant_snippet(doc.text, topic)
                    
                    qa_pair = {
                        "query": query,
                        "expected_answer": generate_expected_answer(topic, doc_type, insurance_type, relevant_snippet),
                        "expected_chunks": [topic, doc_type, insurance_type],
                        "source_doc": source_id,
                        "difficulty": determine_difficulty(topic, doc_type),
                        "doc_type": doc_type,
                        "insurance_type": insurance_type,
                    }
                    qa_pairs.append(qa_pair)
    
    return qa_pairs


def extract_relevant_snippet(text: str, topic: str, max_length: int = 200) -> str:
    """Extract a relevant snippet from text containing the topic."""
    text_lower = text.lower()
    topic_lower = topic.lower()
    
    # Find topic in text
    idx = text_lower.find(topic_lower)
    if idx == -1:
        return ""
    
    # Extract surrounding context
    start = max(0, idx - 100)
    end = min(len(text), idx + max_length)
    
    snippet = text[start:end]
    
    # Clean up snippet
    snippet = re.sub(r'\s+', ' ', snippet).strip()
    
    return snippet


def generate_expected_answer(topic: str, doc_type: str, insurance_type: str, snippet: str) -> str:
    """Generate expected answer based on topic and document content."""
    
    # Topic-specific answer templates
    answer_templates = {
        "knee replacement surgery": {
            "policy_wording": "Joint replacement surgeries including knee replacement may have specific coverage terms. Check the policy wording for waiting periods, sub-limits, and any specific exclusions related to joint replacements.",
            "exclusion_summary": "Joint replacement surgeries may be listed under surgical procedures. Review the exclusions section for specific conditions or waiting periods that apply to knee replacement surgery.",
        },
        "pre-existing diseases": {
            "policy_wording": "Pre-existing diseases are typically covered after a waiting period of 48 months as per IRDAI regulations. Some policies may offer shorter waiting periods. Check the policy schedule for the specific waiting period applicable.",
            "regulation": "As per IRDAI Health Insurance Regulations 2016, pre-existing diseases have a maximum waiting period of 48 months. Some insurers may offer reduced waiting periods. The waiting period credit is transferable on portability.",
        },
        "maternity benefits": {
            "policy_wording": "Maternity benefits typically have a waiting period of 9-24 months. Covered expenses include delivery charges, hospitalization, and newborn care for up to 90 days. Check policy for specific sub-limits and coverage details.",
        },
        "day care procedures": {
            "policy_wording": "Day care procedures (less than 24 hours hospitalization) are covered. Required documents include claim form, doctor's certificate, investigation reports, and hospital authorization.",
        },
        "portability": {
            "regulation": "Under IRDAI Health Insurance Regulations 2016, policyholders can port their health insurance to another insurer. All accumulated benefits including no-claim bonus and waiting period credits are transferable. Request portability at least 45 days before renewal.",
        },
        "total loss": {
            "policy_wording": "For total loss claims, the insurer assesses the pre-accident value (IDV) of the vehicle. Settlement is the IDV minus deductible. Required documents: FIR, vehicle RC, insurance policy copy, and salvage certificate.",
        },
        "no claim bonus": {
            "policy_wording": "No-claim bonus (NCB) is a discount on the own-damage premium for claim-free years. Discounts range from 20% for one year to 50% for five consecutive claim-free years. NCB is transferable between insurers and vehicles.",
        },
    }
    
    # Get specific answer or generate generic one
    if topic in answer_templates:
        if doc_type in answer_templates[topic]:
            return answer_templates[topic][doc_type]
        elif "policy_wording" in answer_templates[topic]:
            return answer_templates[topic]["policy_wording"]
    
    # Generic answer based on snippet
    if snippet:
        return f"Based on the {doc_type}, {snippet[:150]}..."
    
    return f"Refer to the {doc_type} document for specific information about {topic}."


def determine_difficulty(topic: str, doc_type: str) -> str:
    """Determine question difficulty based on topic and doc type."""
    complex_topics = ["portability", "pre-existing diseases", "total loss", "maternity benefits"]
    complex_doc_types = ["regulation", "exclusion_summary"]
    
    if topic in complex_topics or doc_type in complex_doc_types:
        return "hard"
    elif doc_type in ["policy_wording", "network"]:
        return "medium"
    else:
        return "easy"


def generate_eval_set(qa_pairs: List[Dict[str, Any]], max_items: int = 50) -> Dict[str, Any]:
    """
    Generate eval set from Q&A pairs.
    
    Eval set contains test queries with expected keywords for retrieval evaluation.
    """
    # Select diverse set of queries - use query + source_doc as key for more variety
    selected_pairs = []
    seen_queries = set()
    
    for pair in qa_pairs:
        # Use query + source_doc to allow same query from different sources
        query_key = f"{pair['query']}:{pair.get('source_doc', '')}"
        if query_key not in seen_queries and len(selected_pairs) < max_items:
            selected_pairs.append(pair)
            seen_queries.add(query_key)
    
    eval_items = []
    for i, pair in enumerate(selected_pairs, 1):
        eval_items.append({
            "id": f"EVAL-{i:03d}",
            "query": pair["query"],
            "expected_keywords": pair["expected_chunks"],
            "top_k": 3,
            "difficulty": pair["difficulty"],
            "insurance_type": pair.get("insurance_type", "general"),
            "doc_type": pair.get("doc_type", ""),
            "source_doc": pair["source_doc"],
        })
    
    return {
        "project": "claims / insurance",
        "source": "Generated from knowledge base documents in data/knowledge_base",
        "description": "Evaluation set for testing RAG retrieval and answer quality",
        "total_items": len(eval_items),
        "items": eval_items,
    }


def generate_golden_set(qa_pairs: List[Dict[str, Any]], max_items: int = 50) -> Dict[str, Any]:
    """
    Generate golden set from Q&A pairs.
    
    Golden set contains expected answers with citations for comprehensive evaluation.
    """
    # Select diverse set of queries - use query + source_doc as key for more variety
    selected_pairs = []
    seen_queries = set()
    
    for pair in qa_pairs:
        # Use query + source_doc to allow same query from different sources
        query_key = f"{pair['query']}:{pair.get('source_doc', '')}"
        if query_key not in seen_queries and len(selected_pairs) < max_items:
            selected_pairs.append(pair)
            seen_queries.add(query_key)
    
    golden_items = []
    for i, pair in enumerate(selected_pairs, 1):
        golden_items.append({
            "id": f"GOLDEN-{i:03d}",
            "query": pair["query"],
            "expected_answer": pair["expected_answer"],
            "expected_chunks": pair["expected_chunks"],
            "difficulty": pair["difficulty"],
            "insurance_type": pair.get("insurance_type", "general"),
            "doc_type": pair.get("doc_type", ""),
            "source_doc": pair["source_doc"],
            "metadata": {
                "generated_from": "knowledge_base_documents",
                "topic_category": pair["expected_chunks"][0] if pair["expected_chunks"] else "general",
            }
        })
    
    return {
        "project": "claims / insurance",
        "source": "Generated from knowledge base documents in data/knowledge_base",
        "description": "Golden set with expected answers for RAG evaluation",
        "threshold_metrics": {
            "hit_rate_at_5": 0.85,
            "mrr": 0.65,
            "faithfulness": 0.9,
            "answer_correctness": 0.8,
            "llm_judge_avg": 4.0,
            "citation_coverage": 1.0,
        },
        "total_items": len(golden_items),
        "items": golden_items,
    }


def main():
    """Main function to generate eval and golden sets."""
    print("=" * 60)
    print("EVAL AND GOLDEN SET GENERATION")
    print("=" * 60)
    print()
    
    # Step 1: Load documents
    print("Step 1/4: Loading documents from knowledge base...")
    documents = load_documents_from_manifest()
    print(f"  → {len(documents)} document(s) loaded")
    print()
    
    # Step 2: Extract Q&A pairs
    print("Step 2/4: Extracting Q&A pairs from documents...")
    qa_pairs = extract_qa_from_documents(documents)
    print(f"  → {len(qa_pairs)} Q&A pair(s) extracted")
    print()
    
    # Step 3: Generate eval set
    print("Step 3/4: Generating eval set...")
    eval_set = generate_eval_set(qa_pairs, max_items=50)
    eval_path = EVAL_DIR / "eval_set.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)
    print(f"  → Eval set saved to: {eval_path}")
    print(f"  → Total items: {eval_set['total_items']}")
    print()
    
    # Step 4: Generate golden set
    print("Step 4/4: Generating golden set...")
    golden_set = generate_golden_set(qa_pairs, max_items=50)
    golden_path = GOLDEN_DATASET_DIR / "rag_knowledge_base_golden.json"
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, indent=2, ensure_ascii=False)
    print(f"  → Golden set saved to: {golden_path}")
    print(f"  → Total items: {golden_set['total_items']}")
    print()
    
    # Summary
    print("=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Documents processed : {len(documents)}")
    print(f"Q&A pairs extracted : {len(qa_pairs)}")
    print(f"Eval set items      : {eval_set['total_items']}")
    print(f"Golden set items    : {golden_set['total_items']}")
    print()
    
    # Show sample queries
    print("Sample queries from eval set:")
    for item in eval_set['items'][:5]:
        print(f"  - {item['id']}: {item['query'][:70]}...")
    print()
    
    print("Generation complete!")


if __name__ == "__main__":
    main()