"""Simple benchmark harness for the RAG vector backends."""

import json
import math
import os
import shutil
import statistics
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from app.rag.chunkers import Chunk
from app.rag.vectorstores import get_vector_store


def _embedding_for_text(text: str) -> List[float]:
    """Create a deterministic embedding for benchmark purposes."""
    vector = [0.0] * 4
    tokens = [token.lower() for token in text.replace("/", " ").split() if token]
    for index, token in enumerate(tokens):
        vector[index % 4] += (sum(ord(char) for char in token) % 17) / 17.0
    return [round(value, 6) for value in vector]


def _build_sample_chunks() -> List[Chunk]:
    return [
        Chunk(
            text="Hospitalization coverage for acute care and inpatient services.",
            source_id="policy-1",
            source_path="policies/health.md",
            doc_type="policy_wording",
            insurance_type="health",
            insurer="Acme Health",
            product_code="HLT-100",
            claim_type="hospitalization",
            section="Section 1",
            clause_id="1.1",
            raw_metadata={"jurisdiction": "US"},
        ),
        Chunk(
            text="Dental exclusions apply to cosmetic procedures and orthodontics.",
            source_id="policy-2",
            source_path="policies/dental.md",
            doc_type="policy_wording",
            insurance_type="dental",
            insurer="Acme Health",
            product_code="DNT-200",
            claim_type="dental",
            section="Section 2",
            clause_id="2.1",
            raw_metadata={"jurisdiction": "US"},
        ),
        Chunk(
            text="Network agreement covers approved hospitals and emergency services.",
            source_id="network-1",
            source_path="network/agreement.md",
            doc_type="network",
            insurance_type="health",
            insurer="Acme Health",
            product_code="HLT-100",
            claim_type="network",
            section="Section 3",
            clause_id="3.2",
            raw_metadata={"jurisdiction": "US"},
        ),
        Chunk(
            text="Prior memo confirms settlement precedent for outpatient claims.",
            source_id="memo-1",
            source_path="memos/decision.json",
            doc_type="memo",
            insurance_type="health",
            insurer="Acme Health",
            product_code="HLT-100",
            claim_type="settlement",
            section="Appendix",
            clause_id="A.1",
            raw_metadata={"jurisdiction": "US"},
        ),
    ]


def _percentile(values: List[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(percentile_value / 100.0 * len(ordered))) - 1))
    return ordered[index]


def _dir_size(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return total
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.exists(file_path):
                total += os.path.getsize(file_path)
    return total


def benchmark_backend(backend: str) -> Dict[str, Any]:
    temp_dir = tempfile.mkdtemp(prefix=f"{backend}_bench_", dir="/tmp")
    try:
        kwargs: Dict[str, Any] = {}
        if backend == "faiss":
            kwargs["index_path"] = os.path.join(temp_dir, "faiss.index")
        elif backend == "chroma":
            kwargs["persist_directory"] = temp_dir
            kwargs["collection_name"] = f"claims-bench-{backend}"
        elif backend == "pinecone":
            kwargs["index_name"] = f"claims-bench-{backend}"
            kwargs["api_key"] = os.getenv("PINECONE_API_KEY")

        store = get_vector_store(backend=backend, **kwargs)
        chunks = _build_sample_chunks()
        embeddings = [_embedding_for_text(chunk.text) for chunk in chunks]

        ingestion_start = time.perf_counter()
        store.add(chunks, embeddings)
        ingestion_ms = (time.perf_counter() - ingestion_start) * 1000.0
        store.persist()

        queries = [
            ("hospitalization coverage", "policy-1"),
            ("network agreement", "network-1"),
            ("prior memo settlement", "memo-1"),
        ]
        latencies_ms: List[float] = []
        recalls: List[float] = []
        for query_text, expected_id in queries:
            query_embedding = _embedding_for_text(query_text)
            retrieval_start = time.perf_counter()
            results = store.search(query_text, query_embedding, k=5)
            latencies_ms.append((time.perf_counter() - retrieval_start) * 1000.0)
            recalls.append(1.0 if any(result_chunk.source_id == expected_id for result_chunk, _ in results) else 0.0)

        return {
            "backend": backend,
            "status": "ok",
            "ingestion_ms": round(ingestion_ms, 3),
            "retention_latency_ms": {
                "p50": round(_percentile(latencies_ms, 50), 3),
                "p95": round(_percentile(latencies_ms, 95), 3),
            },
            "recall_at_5": round(statistics.mean(recalls), 3),
            "storage_bytes": _dir_size(temp_dir),
            "temp_dir": temp_dir,
        }
    except Exception as exc:  # pragma: no cover - benchmark resilience
        return {
            "backend": backend,
            "status": "error",
            "error": str(exc),
            "temp_dir": temp_dir,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    results = {}
    for backend in ["faiss", "chroma", "pinecone"]:
        results[backend] = benchmark_backend(backend)

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
