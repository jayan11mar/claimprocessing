# Acceptance criteria mapping

This document ties each project requirement to the implemented modules, the tests that cover it, and the evidence artifacts generated for sign-off.

## Ingestion and indexing

| Criterion | Implemented modules | Tests / metrics | Evidence |
| --- | --- | --- | --- |
| Upload documents and create chunks for retrieval | app/api/server.py, app/rag/chunkers.py, app/rag/embeddings.py | tests/test_api_rag_endpoints.py | /ingest and /ingest/status endpoints, plus evaluation evidence in reports/summary.json |
| Build a retrievable index from uploaded documents | app/api/server.py, app/rag/vectorstores.py | tests/test_api_rag_endpoints.py | /retrieve endpoint returns ranked chunks and source_count |

## Retrieval and citations

| Criterion | Implemented modules | Tests / metrics | Evidence |
| --- | --- | --- | --- |
| Return top-k chunks for a user query | app/api/server.py, app/rag/retriever_hybrid.py | tests/test_api_rag_endpoints.py | /retrieve endpoint returns chunk payloads with scores |
| Attach citations to RAG-backed answers | app/api/server.py, app/chains/agent_chain.py | tests/test_api_rag_endpoints.py | /chat surfaces retrieval_trace and citations in the response payload |

## Chat and agent orchestration

| Criterion | Implemented modules | Tests / metrics | Evidence |
| --- | --- | --- | --- |
| Route between deterministic tools and retrieval-backed answers | app/chains/agent_chain.py, app/api/server.py | tests/test_api_rag_endpoints.py | Chat endpoint accepts retrieval metadata and returns structured answers |
| Preserve existing endpoint compatibility while adding the retrieval workflow | app/api/server.py | tests/test_api_rag_and_retrieval.py | Existing health and chat tests remain green alongside the new acceptance coverage |

## Evaluation and sign-off

| Criterion | Implemented modules | Tests / metrics | Evidence |
| --- | --- | --- | --- |
| Produce measurable RAG evaluation outputs | eval/run_eval.py | tests/test_rag_pipeline.py | reports/summary.json and reports/report.md |
| Create a sign-off package for project acceptance | app/rag/acceptance_validation.py | tests/test_rag_pipeline.py | docs/project_signoff_report.md and reports/acceptance_evidence.json |
