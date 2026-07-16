from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.rag.chunkers import Chunk, ChunkConfig
from app.rag.retriever_basic import build_basic_retriever
from app.rag.vectorstores.faiss_store import FAISSStore


def test_faiss_retriever_returns_results_when_embedding_fn_is_provided():
    store = FAISSStore(index_path="/tmp/test_faiss_retriever.index")
    chunk = Chunk(
        text="policy coverage for vehicle damage",
        source_id="policy-1",
        source_path="policy.md",
        doc_type="policy_wording",
        insurance_type="motor",
        product_code="AUTO",
        raw_metadata={"product_name": "Auto Policy"},
    )
    store.add([chunk], [[1.0, 0.0]])

    retriever = store.as_retriever(search_kwargs={"embedding_fn": lambda texts: [[1.0, 0.0] for _ in texts], "k": 1})
    results = retriever.invoke("vehicle damage")

    assert len(results) == 1
    assert results[0].page_content == chunk.text


def test_build_basic_retriever_passes_metadata_filter_to_store(monkeypatch):
    class DummyStore:
        def __init__(self):
            self.search_kwargs = None

        def add(self, chunks, embeddings):
            return None

        def delete(self, ids=None):
            return None

        def persist(self, path=None):
            return None

        def as_retriever(self, search_kwargs=None):
            self.search_kwargs = search_kwargs
            return SimpleNamespace(search_kwargs=search_kwargs)

    dummy_store = DummyStore()

    monkeypatch.setattr("app.rag.retriever_basic.load_documents_from_manifest", lambda: [])
    monkeypatch.setattr("app.rag.retriever_basic.chunk_document", lambda doc, config, use_semantic=True: [])
    monkeypatch.setattr("app.rag.retriever_basic.get_embedding_fn", lambda model_name=None: lambda texts: [[0.0, 1.0] for _ in texts])
    monkeypatch.setattr("app.rag.retriever_basic.get_vector_store", lambda backend="faiss", **kwargs: dummy_store)

    metadata_filter = {"doc_type": "policy_wording", "product_code": "AUTO"}
    retriever = build_basic_retriever(metadata_filter=metadata_filter)

    # embedding_fn is now passed through to the store; verify it's callable
    assert callable(dummy_store.search_kwargs.get("embedding_fn"))
    assert dummy_store.search_kwargs.get("k") == 5
    assert dummy_store.search_kwargs.get("filter") == metadata_filter
    assert getattr(retriever, "search_kwargs", None) is not None
    assert getattr(retriever, "search_kwargs", {}).get("k") == 5
    assert getattr(retriever, "search_kwargs", {}).get("filter") == metadata_filter


def test_settings_reads_chunking_environment(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    monkeypatch.setenv("CHUNK_OVERLAP", "56")
    monkeypatch.setenv("CHUNKING_STRATEGY", "semantic")

    settings = get_settings()

    assert settings.CHUNK_SIZE == 1234
    assert settings.CHUNK_OVERLAP == 56
    assert settings.CHUNKING_STRATEGY == "semantic"

    get_settings.cache_clear()