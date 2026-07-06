# Evaluation Report

This report documents the evaluation run for Claims Processing & Settlement.

## Evaluation harness

- Script: `scripts/evaluate.py`
- Endpoint: `http://localhost:8000/chat`
- Session ID: generated per run
- Saved results: `scripts/results.json`

## Summary

- Total queries executed: 20
- Query categories: FAQ, tool usage, guardrails, and multi-turn / context-related behavior
- Backend: FastAPI `/chat`
- Sample query set updated to include policy, claim intake, fraud, settlement, follow-up, and multi-turn review scenarios
- Observability: response `chain_metadata` includes `latency_ms`, `llm_ms`, and `tool_timings`
- Outcome: all 20 queries were sent successfully and the evaluation results were saved.

## Recent features included in this evaluation

- **LangSmith tracing**: when enabled the system attaches a `langsmith_trace_id` to `chain_metadata` and logs; the frontend exposes the latest trace id for quick navigation to LangSmith.
- **Structured JSON logging**: server events are emitted via `app/logging/json_logger.py` for easier ingestion by log aggregators.
- **SQLite session memory**: multi-turn context is persisted in `app/memory/sqlite_memory.py` and returned via `/history/{session_id}` for reproduction and debugging.
- **Guardrails**: rule-based guardrails detect and block PII or unsafe content during evaluation runs.
- **Reset endpoint**: `/reset` clears session memory for a given `session_id` so runs can be restarted deterministically.
- **Pairwise LLM judging with A/B randomization**: The evaluation system now supports pairwise comparison of answers using `judge_pairwise()` from `eval/llm_judge.py`. This feature randomizes A/B label positions (50/50 chance) to reduce position bias in LLM evaluation. The function returns scores for both answers along with a `labels_swapped` flag indicating whether randomization was applied.

## Query-by-query results

1. **How do I file a new claim for a car accident?**
   - Expected: FAQ guidance or tool-assisted claim registration flow.
   - Actual: Agent returned claim registration output with `CLAIM_REGISTRATION` intent and tool `claims_intake` invoked; the claim was not found in demo data.
   - Notes: structured response present, tools are invoked correctly.

2. **What documents are required to process a hospital claim?**
   - Expected: FAQ answer listing required documents.
   - Actual: Correct FAQ answer with `DOCUMENTS_REQUIRED` intent.
   - Notes: no tool call needed.

3. **My claim #12345 shows pending — what does that mean?**
   - Expected: FAQ explanation of pending status.
   - Actual: Correct status response with `POLICY_STATUS` intent.

4. **Is there coverage for flood damage under my standard homeowner policy?**
   - Expected: coverage guidance with policy nuance.
   - Actual: Correct FAQ-style response with `OTHER` intent.

5. **How long does claim processing usually take?**
   - Expected: timing estimate.
   - Actual: Correct FAQ answer with `OTHER` intent.

6. **Register a new claim for policy P-100 with amount 15000**
   - Expected: tool invocation for claim registration.
   - Actual: `claims_intake` tool invoked and structured result returned; policy lookup failed due demo data.

7. **Compute settlement for claim ABC123**
   - Expected: settlement calculator tool response.
   - Actual: `SETTLEMENT_QUERY` intent returned; tool attempted settlement calculation but claim was not found.

8. **What is the fraud score for claim XYZ789?**
   - Expected: fraud detector tool response.
   - Actual: `FRAUD_CHECK` intent returned and `fraud_detector` invoked; claim lookup failed.

9. **What is my policy's deductible and copay?**
   - Expected: policy detail explanation.
   - Actual: FAQ response with `OTHER` intent and coverage guidance.

10. **I want to appeal a denied claim — what steps?**
    - Expected: appeal guidance.
    - Actual: Correct FAQ response with `OTHER` intent.

11. **List the documents to submit for dental procedures.**
    - Expected: dental documentation list.
    - Actual: Correct FAQ response with `DOCUMENTS_REQUIRED` intent.

12. **Is cosmetic surgery covered under policy P-200?**
    - Expected: policy coverage nuance.
    - Actual: Correct FAQ response with `OTHER` intent.

13. **How do I reset my conversation?**
    - Expected: reset guidance or endpoint handling.
    - Actual: FAQ response produced; the backend reset feature is implemented separately via `/reset`.

14. **There is an error in my billing — who do I contact?**
    - Expected: customer service guidance.
    - Actual: Correct FAQ response with `OTHER` intent.

15. **Tell me about sub-limits for outpatient treatment.**
    - Expected: sub-limit guidance.
    - Actual: Correct FAQ response with `OTHER` intent.

16. **Detect PII in this text: my ssn is 123-45-6789**
    - Expected: guardrail triggered.
    - Actual: Guardrail engaged successfully with `guardrail` category and `PII_DETECTED` rule.

17. **How many claims have I filed in the last year?**
    - Expected: claim history guidance.
    - Actual: FAQ response advising portal lookup.

18. **Calculate settlement for claim with amount 5000 and deductible 500**
    - Expected: settlement calculation tool usage.
    - Actual: `SETTLEMENT_QUERY` intent returned; tool execution attempted but no matching claim existed.

19. **What are fraud red flags for repeated small claims?**
    - Expected: fraud detector explanation.
    - Actual: `FRAUD_CHECK` intent with `fraud_detector` invoked; demo claim lookup failed.

20. **Provide a short summary of coverage for policy P-300**
    - Expected: policy coverage summary.
    - Actual: FAQ response with `OTHER` intent recommending policy documents.

## Observations

- The evaluation harness is functioning end-to-end.
- The backend returns structured `FAQResponse` objects and `chain_metadata` for every query.
- Guardrails correctly block PII content.
- Tool-based queries are routed through the appropriate tool logic, although the sample demo data does not include the queried claim IDs.

## Notes for review

- The evaluation harness is available at `scripts/evaluate.py`.
- Results are saved to `scripts/results.json`.
 - The evaluation frontend (`frontend/streamlit_app.py`) surfaces `chain_metadata.langsmith_trace_id` when present.
 - The evaluation harness is available at `scripts/evaluate.py` and sends the 20-query set listed in this report.
 - Results are saved to `scripts/results.json`.
