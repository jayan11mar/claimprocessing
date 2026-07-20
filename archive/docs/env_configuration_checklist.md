# Environment configuration checklist

Update these values in your local environment files before running the RAG evaluation harness.

- RAG_EVALUATION_CONTEXT: selects the acceptance profile to use. Supported values: loan underwriting, customer svc, aml / fraud, claims / insurance.
- RAG_EVALUATION_HIT_RATE_THRESHOLD: minimum Hit Rate @ 5 required for a case to pass.
- RAG_EVALUATION_MRR_THRESHOLD: minimum MRR required for a case to pass.
- RAG_EVALUATION_FAITHFULNESS_THRESHOLD: minimum faithfulness score required for a case to pass.
- RAG_EVALUATION_ANSWER_CORRECTNESS_THRESHOLD: minimum answer correctness score required for a case to pass.
- RAG_EVALUATION_LLM_JUDGE_AVG_THRESHOLD: minimum LLM judge average (/5) required for a case to pass.
- RAG_EVALUATION_CITATION_COVERAGE_THRESHOLD: minimum citation coverage required for a case to pass.
- RAG_EVALUATION_MIN_CITATIONS: minimum number of citations a response should include.
- RAG_EVALUATION_OUTPUT_PATH: optional file path for saving evaluation JSON results.

Use the same values in `.env` and `.env.example` so local runs and CI use consistent acceptance thresholds.
