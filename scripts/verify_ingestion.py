"""
Verification script for knowledge base ingestion.
Checks:
1. FAISS index contents (chunk count, dimension)
2. Metadata JSON file consistency (chunks match FAISS vectors)
3. Source IDs match manifest.yaml
4. Unique insurance_type and doc_type values
5. Day care / daycare keyword presence in chunks
6. Cross-reference: manifest sources vs loaded documents vs FAISS vectors

Usage:
    python scripts/verify_ingestion.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss

from app.rag.loaders import load_manifest, load_documents_from_manifest
from app.rag.chunkers import chunk_document, ChunkConfig
from app.rag.vectorstores.faiss_store import FAISSStore


def verify_faiss_index():
    """Check the persisted FAISS index file."""
    index_path = Path(__file__).parent.parent / "data" / "faiss_index"
    print("=" * 60)
    print("FAISS INDEX VERIFICATION")
    print("=" * 60)

    if not index_path.exists():
        print(f"  [FAIL] FAISS index not found at: {index_path}")
        print("  Run: python -m app.rag.ingest_basic")
        return None

    try:
        index = faiss.read_index(str(index_path))
        print(f"  [OK]   Index file exists: {index_path}")
        print(f"  [OK]   Index type: {type(index).__name__}")
        print(f"  [OK]   Embedding dimension: {index.d}")
        print(f"  [OK]   Total vectors stored: {index.ntotal}")
        return index
    except Exception as e:
        print(f"  [FAIL] Could not read FAISS index: {e}")
        return None


def verify_metadata_json(index_ntotal):
    """Check the metadata JSON file and cross-reference with FAISS vectors."""
    meta_path = Path(__file__).parent.parent / "data" / "faiss_index.meta.json"
    print("\n" + "=" * 60)
    print("METADATA JSON VERIFICATION")
    print("=" * 60)

    if not meta_path.exists():
        print(f"  [FAIL] Metadata JSON not found at: {meta_path}")
        print("  Run: python -m app.rag.ingest_basic")
        return None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"  [FAIL] Could not read metadata JSON: {e}")
        return None

    chunks = meta.get("chunks", [])
    chunk_ids = meta.get("chunk_ids", [])
    embedding_version = meta.get("embedding_model_version", "unknown")

    print(f"  [OK]   Metadata JSON exists: {meta_path}")
    print(f"  [OK]   Chunks in metadata: {len(chunks)}")
    print(f"  [OK]   Chunk IDs: {len(chunk_ids)}")
    print(f"  [OK]   Embedding model version: {embedding_version}")

    if index_ntotal is not None:
        if len(chunks) == index_ntotal:
            print(f"  [PASS] Metadata count ({len(chunks)}) == FAISS vectors ({index_ntotal}) ✓")
        else:
            print(f"  [FAIL] Metadata count ({len(chunks)}) != FAISS vectors ({index_ntotal}) ✗")
            print("         Re-run ingestion to sync: python -m app.rag.ingest_basic")

    return chunks


def verify_manifest_sources():
    """List all sources defined in the manifest."""
    print("\n" + "=" * 60)
    print("MANIFEST SOURCES")
    print("=" * 60)

    manifest = load_manifest()
    sources = manifest.get("sources", [])
    print(f"  Total sources defined: {len(sources)}")
    print()
    for s in sources:
        path_val = s.get("path", "?")
        exists = "✓" if (Path(__file__).parent.parent / "data" / "knowledge_base" / path_val).exists() else "✗ MISSING"
        print(f"  [{s.get('doc_type', '?'):20s}] {s.get('id', '?'):35s} → {path_val:50s} {exists}")
    return sources


def verify_document_chunking():
    """Load documents, chunk them, and show per-document breakdown."""
    print("\n" + "=" * 60)
    print("DOCUMENT CHUNKING VERIFICATION")
    print("=" * 60)

    documents = load_documents_from_manifest()
    print(f"  Documents loaded: {len(documents)}")
    print()

    chunk_config = ChunkConfig(chunk_size=800, chunk_overlap=100)
    total_chunks = 0

    for doc in documents:
        chunks = chunk_document(doc, chunk_config, use_semantic=True)
        total_chunks += len(chunks)
        print(f"  [{doc.doc_type:20s}] {doc.source_id:35s} → {len(chunks):4d} chunks  "
              f"(text_len={len(doc.text):6d})")

    print(f"\n  {'TOTAL':57s} → {total_chunks:4d} chunks")
    return total_chunks


def verify_chunk_metadata(chunks_meta):
    """Analyze metadata from the persisted index."""
    print("\n" + "=" * 60)
    print("CHUNK METADATA ANALYSIS")
    print("=" * 60)

    if not chunks_meta:
        print("  [FAIL] No chunks to analyze")
        return

    # Source IDs
    source_ids = set(c.get("source_id", "?") for c in chunks_meta)
    print(f"  Unique source_ids ({len(source_ids)}):")
    for sid in sorted(source_ids):
        count = sum(1 for c in chunks_meta if c.get("source_id") == sid)
        print(f"    {sid:40s} ({count} chunks)")

    # Insurance types
    insurance_types = set(c.get("insurance_type", "?") for c in chunks_meta)
    print(f"\n  Unique insurance_type values: {insurance_types}")
    for it in sorted(insurance_types):
        count = sum(1 for c in chunks_meta if c.get("insurance_type") == it)
        print(f"    {it:10s}: {count} chunks")

    # Doc types
    doc_types = set(c.get("doc_type", "?") for c in chunks_meta)
    print(f"\n  Unique doc_type values: {doc_types}")
    for dt in sorted(doc_types):
        count = sum(1 for c in chunks_meta if c.get("doc_type") == dt)
        print(f"    {dt:25s}: {count} chunks")

    # Check for dummy source IDs
    dummy_ids = [sid for sid in source_ids if sid in ("policy-1", "policy-week6")]
    if dummy_ids:
        print(f"\n  [FAIL] Dummy/test source_ids found: {dummy_ids}")
        print("         The index was built from test data, not from manifest.yaml!")
    else:
        print(f"\n  [PASS] No dummy/test source_ids found ✓")

    # Check source IDs match manifest patterns
    manifest = load_manifest()
    manifest_ids = set(s.get("id") for s in manifest.get("sources", []))
    missing_in_manifest = [sid for sid in source_ids if sid not in manifest_ids]
    if missing_in_manifest:
        print(f"\n  [WARN] Source IDs not in manifest.yaml: {missing_in_manifest}")
    else:
        print(f"  [PASS] All source_ids match manifest.yaml ✓")

    # Check coverage of manifest IDs
    found_in_index = [mid for mid in manifest_ids if mid in source_ids]
    missed_in_index = [mid for mid in manifest_ids if mid not in source_ids]
    if missed_in_index:
        print(f"\n  [WARN] Manifest source_ids NOT found in index:")
        for mid in missed_in_index:
            print(f"    {mid}")
    else:
        print(f"  [PASS] All manifest source_ids present in index ✓")


def verify_daycare_keywords(chunks_meta):
    """Check for day care related keywords in chunk text."""
    print("\n" + "=" * 60)
    print("DAY CARE KEYWORD CHECK")
    print("=" * 60)

    if not chunks_meta:
        print("  [FAIL] No chunks to search")
        return

    patterns = [
        ("day care", "day care"),
        ("daycare", "daycare"),
        ("day care procedure", "day care procedure"),
        ("day care treatment", "day care treatment"),
        ("daycare treatment", "daycare treatment"),
    ]

    for label, pattern in patterns:
        matching = []
        for i, c in enumerate(chunks_meta):
            text = c.get("text", "").lower()
            if pattern.lower() in text:
                matching.append(c)
        print(f"\n  '{label}' keyword:")
        if matching:
            print(f"    Found in {len(matching)} chunk(s):")
            for c in matching[:5]:
                source_id = c.get("source_id", "?")
                insurance_type = c.get("insurance_type", "?")
                chunk_idx = c.get("chunk_index", "?")
                text_preview = c.get("text", "")[:120].replace("\n", " ")
                print(f"    └─ {source_id}[{chunk_idx}] ({insurance_type}): {text_preview}...")
            if len(matching) > 5:
                print(f"    ... and {len(matching) - 5} more")
        else:
            print("    Not found in any chunk")


def verify_health_motor_coverage(chunks_meta):
    """Verify both health and motor insurance types are present if in manifest."""
    print("\n" + "=" * 60)
    print("INSURANCE TYPE COVERAGE CHECK")
    print("=" * 60)

    if not chunks_meta:
        print("  [FAIL] No chunks to analyze")
        return

    manifest = load_manifest()
    manifest_insurance_types = set(s.get("insurance_type") for s in manifest.get("sources", []))
    chunk_insurance_types = set(c.get("insurance_type") for c in chunks_meta if c.get("insurance_type"))

    missing_types = manifest_insurance_types - chunk_insurance_types
    extra_types = chunk_insurance_types - manifest_insurance_types

    print(f"  Insurance types in manifest: {manifest_insurance_types}")
    print(f"  Insurance types in index:   {chunk_insurance_types}")

    if missing_types:
        print(f"  [WARN] Missing insurance types from index: {missing_types}")
    else:
        print(f"  [PASS] All manifest insurance types present ✓")

    if extra_types:
        print(f"  [INFO] Extra insurance types in index (not in manifest): {extra_types}")


def main():
    print("=" * 70)
    print("KNOWLEDGE BASE INGESTION VERIFICATION")
    print("=" * 70)

    # Step 1: Verify FAISS index
    index = verify_faiss_index()
    index_ntotal = index.ntotal if index else None

    # Step 2: Verify metadata JSON
    chunks_meta = verify_metadata_json(index_ntotal)

    # Step 3: Verify manifest sources
    sources = verify_manifest_sources()

    # Step 4: Verify chunking produces expected number of chunks
    chunk_count = verify_document_chunking()

    # Step 5: Analyze chunk metadata
    verify_chunk_metadata(chunks_meta)

    # Step 6: Day care keyword check
    verify_daycare_keywords(chunks_meta)

    # Step 7: Health/motor coverage check
    verify_health_motor_coverage(chunks_meta)

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Manifest sources          : {len(sources)}")
    print(f"  Documents loaded          : {len(load_documents_from_manifest())}")
    print(f"  Chunks created (manifest) : {chunk_count}")
    if chunks_meta:
        print(f"  Chunks persisted          : {len(chunks_meta)}")
    if index:
        print(f"  FAISS vectors stored      : {index.ntotal}")

    if chunks_meta and index_ntotal:
        if len(chunks_meta) == index_ntotal:
            print(f"  [PASS] Metadata count == FAISS vectors ✓")
        else:
            print(f"  [FAIL] Metadata count ({len(chunks_meta)}) != FAISS vectors ({index_ntotal}) ✗")

    # Check for dummy data
    if chunks_meta:
        dummy_ids = [c.get("source_id") for c in chunks_meta if c.get("source_id") in ("policy-1", "policy-week6")]
        if dummy_ids:
            print(f"  [FAIL] Dummy source_ids detected: {set(dummy_ids)}. RE-INGEST REQUIRED!")
        else:
            print(f"  [PASS] No dummy/test data in index ✓")

    print("=" * 70)


if __name__ == "__main__":
    main()