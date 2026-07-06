"""Shared metadata schema helpers for RAG chunks."""

from typing import Any, Dict, List

REQUIRED_METADATA_FIELDS: List[str] = [
    "doc_type",
    "insurance_type",
    "insurer",
    "product_code",
    "claim_type",
    "section",
    "clause_id",
]


def build_chunk_metadata(chunk: Any) -> Dict[str, Any]:
    """Build a metadata dictionary for a chunk using the shared schema.

    Required fields from the shared schema are attached first, followed by any
    additional values stored in the chunk's raw metadata.
    """
    metadata: Dict[str, Any] = {}

    for field_name in REQUIRED_METADATA_FIELDS:
        value = getattr(chunk, field_name, None)
        if value is not None and value != "":
            metadata[field_name] = value

    for key, value in getattr(chunk, "raw_metadata", {}).items():
        if key in metadata:
            continue
        if value is None or value == "":
            continue
        metadata[key] = value

    return metadata
