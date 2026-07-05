# Vector backend choice

This project now supports a pluggable vector backend layer for the claims knowledge base.

## Recommended default for local development

FAISS is the default backend for local development because it is dependency-light, runs fully offline, and fits the lightweight evaluation workflow in this repository.

## Benchmark notes

Run the benchmark script with:

```bash
python -m app.rag.benchmarks.vector_backend_bench
```

The script reports ingestion latency, p50/p95 retrieval latency, recall@5, and storage footprint for the available backends. A Pinecone run is only attempted when the Pinecone SDK and API key are configured.

### Local benchmark snapshot

Observed locally on the sample corpus:

- FAISS: ingestion 0.107 ms, p50 retrieval 0.019 ms, p95 retrieval 0.457 ms, recall@5 1.0, storage 109 bytes.
- Chroma: ingestion 238.905 ms, p50 retrieval 1.113 ms, p95 retrieval 2.82 ms, recall@5 1.0, storage 204,516 bytes.
- Pinecone: skipped because PINECONE_API_KEY is not configured in this environment.

## Selection rationale

- FAISS: best fit for local experiments and fast iteration.
- Chroma: preferred when persistent local collections and a simple metadata layer are desired.
- Pinecone: best for managed production deployments, but it requires external configuration.
