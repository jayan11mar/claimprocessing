# RAG Evaluation Report - Failure Analysis

## Summary

- Projects evaluated: 1
- Cases evaluated: 10
- Passed cases: 1
- Failed cases: 9

## Failure Buckets

- Retrieval: 4 cases
- Answer quality: 5 cases
- Citations: 0 cases
- Other: 0 cases

## Detailed Failure Analysis

### Top Failed Queries

#### 1. FAIL-01: What is the claim settlement time for motor insurance?

**Failure Reason:** missing evidence - retrieval system returned completely irrelevant documents

**Metrics:**
- Hit@K: 0.0 (threshold: 0.85)
- MRR: 0.0 (threshold: 0.65)
- Faithfulness: 1.0 (threshold: 0.9)
- Answer Correctness: 1.0 (threshold: 0.8)
- LLM Judge Score: 5.0 (threshold: 0.8)

**Source:** synthetic failure case - retrieval failure

---

#### 2. FAIL-03: What documents are needed for cashless hospitalization?

**Failure Reason:** missing evidence - no chunks retrieved (empty retrieval)

**Metrics:**
- Hit@K: 0.0 (threshold: 0.85)
- MRR: 0.0 (threshold: 0.65)
- Faithfulness: 1.0 (threshold: 0.9)
- Answer Correctness: 1.0 (threshold: 0.8)
- LLM Judge Score: 3.75 (threshold: 0.8)

**Source:** synthetic failure case - empty retrieval

---

#### 3. FAIL-04: How much does a no-claim bonus reduce my motor insurance premium?

**Failure Reason:** answer quality - completely incorrect information about NCB

**Metrics:**
- Hit@K: 0.0 (threshold: 0.85)
- MRR: 0.0 (threshold: 0.65)
- Faithfulness: 0.136 (threshold: 0.9)
- Answer Correctness: 0.136 (threshold: 0.8)
- LLM Judge Score: 1.715 (threshold: 0.8)

**Source:** synthetic failure case - wrong answer

---

#### 4. FAIL-05: What is the IRDAI regulation on health insurance portability?

**Failure Reason:** answer quality - partially correct but contains critical misinformation about portability benefits

**Metrics:**
- Hit@K: 0.0 (threshold: 0.85)
- MRR: 0.0 (threshold: 0.65)
- Faithfulness: 0.333 (threshold: 0.9)
- Answer Correctness: 0.333 (threshold: 0.8)
- LLM Judge Score: 2.208 (threshold: 0.8)

**Source:** synthetic failure case - partially incorrect answer

---

#### 5. FAIL-06: What are the fraud indicators in motor insurance claims?

**Failure Reason:** answer quality - answer contradicts expected fraud indicators (inverted logic)

**Metrics:**
- Hit@K: 0.0 (threshold: 0.85)
- MRR: 0.0 (threshold: 0.65)
- Faithfulness: 0.588 (threshold: 0.9)
- Answer Correctness: 0.588 (threshold: 0.8)
- LLM Judge Score: 2.845 (threshold: 0.8)

**Source:** synthetic failure case - hallucinated/contradictory answer

---

#### 6. FAIL-07: Can I get reimbursement for treatment at a non-network hospital?

**Failure Reason:** citations - answer contradicts retrieved context which mentions non-network coverage

**Metrics:**
- Hit@K: 1.0 (threshold: 0.85)
- MRR: 1.0 (threshold: 0.65)
- Faithfulness: 0.556 (threshold: 0.9)
- Answer Correctness: 0.161 (threshold: 0.8)
- LLM Judge Score: 3.027 (threshold: 0.8)

**Source:** synthetic failure case - answer contradicts evidence

---

#### 7. FAIL-08: What is the waiting period for maternity benefits in health insurance?

**Failure Reason:** citations - answer not grounded in retrieved context which mentions waiting period

**Metrics:**
- Hit@K: 1.0 (threshold: 0.85)
- MRR: 1.0 (threshold: 0.65)
- Faithfulness: 0.923 (threshold: 0.9)
- Answer Correctness: 0.29 (threshold: 0.8)
- LLM Judge Score: 3.35 (threshold: 0.8)

**Source:** synthetic failure case - answer not supported by evidence

---

#### 8. FAIL-09: How does the deductible work in motor insurance claims?

**Failure Reason:** missing evidence - poor retrieval combined with incorrect answer (inverted deductible explanation)

**Metrics:**
- Hit@K: 1.0 (threshold: 0.85)
- MRR: 1.0 (threshold: 0.65)
- Faithfulness: 1.0 (threshold: 0.9)
- Answer Correctness: 0.63 (threshold: 0.8)
- LLM Judge Score: 4.2 (threshold: 0.8)

**Source:** synthetic failure case - combined retrieval and answer failure

---

#### 9. FAIL-10: What should I do if the TPA delays my cashless authorization?

**Failure Reason:** missing evidence - poor retrieval plus completely wrong answer denying escalation options

**Metrics:**
- Hit@K: 1.0 (threshold: 0.85)
- MRR: 1.0 (threshold: 0.65)
- Faithfulness: 0.333 (threshold: 0.9)
- Answer Correctness: 0.147 (threshold: 0.8)
- LLM Judge Score: 2.993 (threshold: 0.8)

**Source:** synthetic failure case - combined retrieval and answer failure

---

