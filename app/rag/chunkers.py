"""
Chunking strategies for the claims knowledge base.
Implements recursive and semantic chunking with metadata preservation.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class ChunkConfig:
    """Configuration for chunking operations."""
    chunk_size: int = 800
    chunk_overlap: int = 100
    separators: Optional[List[str]] = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    source_id: str
    source_path: str
    doc_type: str
    insurance_type: str
    product_code: Optional[str] = None
    product_name: Optional[str] = None
    claim_type: Optional[str] = None
    section: Optional[str] = None
    clause_id: Optional[str] = None
    chunk_index: int = 0
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "doc_type": self.doc_type,
            "insurance_type": self.insurance_type,
            "product_code": self.product_code,
            "product_name": self.product_name,
            "claim_type": self.claim_type,
            "section": self.section,
            "clause_id": self.clause_id,
            "chunk_index": self.chunk_index,
            "raw_metadata": self.raw_metadata,
        }


def recursive_chunk(
    text: str,
    config: Optional[ChunkConfig] = None,
    **metadata
) -> List[Chunk]:
    """
    Split text using RecursiveCharacterTextSplitter with Week-6 defaults.

    Args:
        text: The text to chunk.
        config: Chunk configuration (size ~800, overlap ~100).
        **metadata: Metadata to attach to each chunk.

    Returns:
        List of Chunk objects.
    """
    if config is None:
        config = ChunkConfig()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
    )

    texts = splitter.split_text(text)
    chunks = []

    for i, chunk_text in enumerate(texts):
        chunks.append(Chunk(
            text=chunk_text,
            chunk_index=i,
            **metadata
        ))

    return chunks


def _extract_section_info(text: str) -> List[Dict[str, Any]]:
    """
    Extract section and clause information from text.
    Looks for common patterns like "Section X", "Clause X.Y.Z", "Article X".

    Args:
        text: The text to analyze.

    Returns:
        List of section/clause markers found.
    """
    sections = []

    # Pattern for section headings
    section_patterns = [
        r"(?i)(section|clause|article)\s+([0-9]+(?:\.[0-9]+)*)\s*[:\-\.]?\s*(.*?)(?=\n|$)",
        r"(?i)^([0-9]+(?:\.[0-9]+)*)\s*\n\s*(.+?)(?=\n\n|$)",
    ]

    for pattern in section_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            sections.append({
                "full_match": match.group(0),
                "clause_id": match.group(1) if match.lastindex >= 1 else None,
                "section": match.group(2) if match.lastindex >= 2 else None,
            })

    return sections


def semantic_chunk(
    text: str,
    config: Optional[ChunkConfig] = None,
    **metadata
) -> List[Chunk]:
    """
    Semantic chunker for policy/regulation/network docs.
    Respects headings, clause numbering, and section breaks.

    Args:
        text: The text to chunk.
        config: Chunk configuration.
        **metadata: Metadata to attach to each chunk.

    Returns:
        List of Chunk objects with section/clause_id where available.
    """
    if config is None:
        config = ChunkConfig()

    # First, try to split by major section breaks
    section_breaks = re.split(r"\n\s*\n(?=[A-Z]|\d+[\.\s])", text)

    chunks = []
    chunk_index = 0

    for section_text in section_breaks:
        if not section_text.strip():
            continue

        # Extract section/clause info from this section
        section_info = _extract_section_info(section_text)

        # If section is small enough, keep as single chunk
        if len(section_text) <= config.chunk_size:
            clause_id = None
            section = None
            if section_info:
                clause_id = section_info[0].get("clause_id")
                section = section_info[0].get("section")

            chunks.append(Chunk(
                text=section_text,
                section=section,
                clause_id=clause_id,
                chunk_index=chunk_index,
                **metadata
            ))
            chunk_index += 1
        else:
            # Apply recursive chunking within this section
            section_chunks = recursive_chunk(
                section_text,
                config,
                **metadata
            )

            for sc in section_chunks:
                # Try to extract clause info from the chunk
                chunk_section_info = _extract_section_info(sc.text)
                if chunk_section_info:
                    sc.clause_id = chunk_section_info[0].get("clause_id")
                    sc.section = chunk_section_info[0].get("section")
                sc.chunk_index = chunk_index
                chunks.append(sc)
                chunk_index += 1

    return chunks


def chunk_document(
    document: Any,
    config: Optional[ChunkConfig] = None,
    use_semantic: bool = True
) -> List[Chunk]:
    """
    Chunk a document based on its type.
    Uses semantic chunking for policy/regulation/network docs,
    and recursive chunking for other types.

    Args:
        document: Document object to chunk.
        config: Chunk configuration.
        use_semantic: Whether to use semantic chunking for structured docs.

    Returns:
        List of Chunk objects.
    """
    if config is None:
        config = ChunkConfig()

    # Determine if we should use semantic chunking
    semantic_types = {"policy_wording", "regulation", "network_agreement", "exclusion_summary"}

    if use_semantic and document.doc_type in semantic_types:
        return semantic_chunk(
            document.text,
            config,
            source_id=document.source_id,
            source_path=document.source_path,
            doc_type=document.doc_type,
            insurance_type=document.insurance_type,
            product_code=document.product_code,
            product_name=document.product_name,
            claim_type=document.claim_type,
            raw_metadata=document.raw_metadata,
        )
    else:
        return recursive_chunk(
            document.text,
            config,
            source_id=document.source_id,
            source_path=document.source_path,
            doc_type=document.doc_type,
            insurance_type=document.insurance_type,
            product_code=document.product_code,
            product_name=document.product_name,
            claim_type=document.claim_type,
            raw_metadata=document.raw_metadata,
        )