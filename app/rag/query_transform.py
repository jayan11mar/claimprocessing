"""Query transformation helpers for hybrid retrieval."""

import re
from typing import List, Optional


def expand_query(query: str, context: Optional[str] = None) -> List[str]:
    """Expand a user query into a small set of retrieval variants."""
    base_query = re.sub(r"\s+", " ", (query or "").strip())
    if not base_query:
        return []

    variants = [base_query]
    variants.append(f"{base_query} policy clause")
    variants.append(f"{base_query} exclusions")
    variants.append(f"{base_query} evidence")

    if context:
        variants.append(f"{base_query} {context}")

    return list(dict.fromkeys(variants))


def build_query_variants(query: str, context: Optional[str] = None) -> List[str]:
    """Compatibility wrapper used by the hybrid retriever."""
    return expand_query(query, context)
