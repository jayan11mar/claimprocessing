# Knowledge Base Eval and Golden Sets

This document describes the evaluation and golden sets generated from the knowledge base documents in `data/knowledge_base`.

## Overview

The eval and golden sets were automatically generated to test RAG (Retrieval-Augmented Generation) performance on the insurance knowledge base. These datasets help identify retrieval issues and ensure answer quality.

## Generated Files

### 1. Eval Set: `eval/eval_set.json`
- **Purpose**: Test queries for evaluating retrieval quality
- **Items**: 26 test queries
- **Structure**:
  - `query`: The test question
  - `expected_keywords`: Keywords that should appear in retrieved chunks
  - `top_k`: Number of chunks to retrieve (default: 3)
  - `difficulty`: easy/medium/hard
  - `insurance_type`: health/motor/general
  - `doc_type`: policy_wording/regulation/exclusion_summary/network/memo
  - `source_doc`: Source document ID

### 2. Golden Set: `data/golden_dataset/rag_knowledge_base_golden.json`
- **Purpose**: Expected answers with citations for comprehensive evaluation
- **Items**: 26 Q&A pairs
- **Structure**:
  - `query`: The test question
  - `expected_answer`: The expected answer based on document content
  - `expected_chunks`: Keywords that should be in retrieved chunks
  - `difficulty`: easy/medium/hard
  - `insurance_type`: health/motor/general
  - `doc_type`: policy_wording/regulation/exclusion_summary/network/memo
  - `source_doc`: Source document ID
  - `metadata`: Additional context (topic_category, generation info)

## Generation Process

The datasets were generated using `scripts/generate_eval_golden_sets.py`:

1. **Load Documents**: Loads all documents from `data/knowledge_base` using the manifest
2. **Extract Q&A Pairs**: Analyzes document content and generates relevant questions
3. **Generate Eval Set**: Creates test queries with expected keywords
4. **Generate Golden Set**: Creates expected answers based on document content

### Document Sources

The knowledge base contains:
- **Health Policies** (3): HDFC ERGO, Kotak Mahindra, SBI
- **Motor Policies** (2): SBI Private Car, SBI Motor
- **Regulations** (1): IRDAI Health Insurance Regulations 2016
- **Network Agreements** (2): Hospital network agreements
- **Exclusions** (1): Health exclusions summary
- **Adjudication Memos** (1): Prior adjudication memos

**Total**: 59 documents processed, 46 Q&A pairs extracted, 26 unique items in each set.

## Dataset Statistics

### Eval Set
- **Total Items**: 26
- **Difficulty Distribution**:
  - Easy: 0 (0%)
  - Medium: 18 (69%)
  - Hard: 8 (31%)
- **Insurance Type Distribution**:
  - Health: 20 (77%)
  - Motor: 6 (23%)
- **Document Type Distribution**:
  - Policy Wording: 20 (77%)
  - Regulation: 4 (15%)
  - Network: 2 (8%)

### Golden Set
- **Total Items**: 26
- **Difficulty Distribution**:
  - Easy: 0 (0%)
  - Medium: 18 (69%)
  - Hard: 8 (31%)
- **Same distribution as eval set**

## Usage

### Running Evaluation

Use the eval set with the RAG evaluation harness:

```python
from app.rag.evaluation_harness import evaluate_rag_queries
import json

# Load eval set
with open("eval/eval_set.json", "r") as f:
    eval_set = json.load(f)

# Run evaluation
results = evaluate_rag_queries(cases=eval_set["items"])

# Check results
print(f"Passed: {results['summary']['passed_cases']}/{results['summary']['total_cases']}")
```

### Using Golden Set

The golden set provides expected answers for comprehensive testing:

```python
from app.rag.evaluation_harness import run_rag_evaluation

# Load golden set
with open("data/golden_dataset/rag_knowledge_base_golden.json", "r") as f:
    golden_set = json.load(f)

# Run evaluation with golden set
report = run_rag_evaluation(cases=golden_set["items"])

# Analyze results
for case in report["cases"]:
    print(f"Query: {case['query']}")
    print(f"Expected: {case['expected_keywords']}")
    print(f"Retrieved: {case['retrieval_score']:.3f}")
    print(f"Answer Score: {case['answer_score']:.3f}")
    print()
```

### Regenerating Datasets

To regenerate the datasets after updating the knowledge base:

```bash
python scripts/generate_eval_golden_sets.py
```

### Validating Datasets

To validate the generated datasets:

```bash
python scripts/validate_generated_datasets.py
```

## Addressing the Knee Replacement Issue

The original issue reported incorrect RAG responses for knee replacement surgery queries. The generated datasets include specific test cases for this scenario:

**Query**: "Are there any exclusions for knee replacement surgery? I have a health insurance policy from SBI."

**Expected Keywords**: `["knee replacement surgery", "policy_wording", "health"]`

**Expected Answer**: "Joint replacement surgeries including knee replacement may have specific coverage terms. Check the policy wording for waiting periods, sub-limits, and any specific exclusions related to joint replacements."

This test case helps ensure the RAG system correctly retrieves and answers questions about surgical procedure exclusions.

## Threshold Metrics

The golden set includes recommended threshold metrics for evaluation:

```json
{
  "hit_rate_at_5": 0.85,      // 85% of relevant chunks in top 5
  "mrr": 0.65,                // Mean Reciprocal Rank >= 0.65
  "faithfulness": 0.9,        // 90% faithfulness to source
  "answer_correctness": 0.8,  // 80% answer correctness
  "llm_judge_avg": 4.0,       // Average LLM judge score >= 4.0
  "citation_coverage": 1.0    // 100% citation coverage
}
```

## Topics Covered

### Health Insurance
- Pre-existing diseases
- Maternity benefits
- Day care procedures
- Pre-hospitalization
- Post-hospitalization
- ICU charges
- AYUSH treatments
- Newborn coverage
- Senior citizen
- Knee replacement surgery
- Portability

### Motor Insurance
- Total loss
- Third party liability
- Own damage
- Deductible
- No claim bonus
- Flood damage

### General Topics
- Claim settlement
- Network hospitals
- Cashless hospitalization
- Reimbursement claims

## Integration with Existing Tests

The generated datasets integrate with existing RAG tests:

- `tests/test_rag_golden_dataset.py`: Tests against golden dataset
- `tests/test_knowledge_retrieval_integration.py`: Integration tests
- `app/rag/evaluation_harness.py`: Evaluation framework

## Next Steps

1. **Run Baseline Evaluation**: Use the eval set to establish baseline metrics
2. **Identify Issues**: Use the golden set to identify specific failure cases
3. **Improve Retrieval**: Adjust chunking, embedding, or retrieval strategies
4. **Re-evaluate**: Regenerate and re-run evaluation to measure improvements
5. **Expand Coverage**: Add more test cases for edge cases and specific scenarios

## Files Reference

- **Generation Script**: `scripts/generate_eval_golden_sets.py`
- **Validation Script**: `scripts/validate_generated_datasets.py`
- **Eval Set**: `eval/eval_set.json`
- **Golden Set**: `data/golden_dataset/rag_knowledge_base_golden.json`
- **Knowledge Base**: `data/knowledge_base/`
- **Manifest**: `data/knowledge_base/manifest.yaml`

## Notes

- The datasets are automatically generated based on document content analysis
- Some expected answers use template responses for common topics
- The datasets focus on retrieval quality rather than conversational aspects
- Difficulty levels are assigned based on topic complexity and document type
- The datasets complement existing golden datasets (rag_claims_insurance.json, etc.)