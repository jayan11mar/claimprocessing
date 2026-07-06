from app.rag.chunkers import Chunk
from app.rag.metadata import REQUIRED_METADATA_FIELDS, build_chunk_metadata
from app.rag.vectorstores import get_vector_store
from app.rag.vectorstores.faiss_store import FAISSStore


def test_factory_returns_faiss_store_for_faiss_backend():
    store = get_vector_store("faiss")
    assert isinstance(store, FAISSStore)


def test_build_chunk_metadata_includes_required_fields():
    chunk = Chunk(
        text="Coverage details",
        source_id="policy-1",
        source_path="policies/health.md",
        doc_type="policy_wording",
        insurance_type="health",
        insurer="Acme Health",
        product_code="HLT-100",
        claim_type="hospitalization",
        section="Section 1",
        clause_id="1.2",
        raw_metadata={"jurisdiction": "US"},
    )

    metadata = build_chunk_metadata(chunk)

    assert set(REQUIRED_METADATA_FIELDS).issubset(metadata.keys())
    assert metadata["doc_type"] == "policy_wording"
    assert metadata["insurer"] == "Acme Health"
    assert metadata["jurisdiction"] == "US"
