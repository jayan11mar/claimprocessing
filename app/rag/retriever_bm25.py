"""A lightweight BM25 retriever for claims knowledge chunks."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

from app.rag.chunkers import Chunk


@dataclass
class BM25Result:
    """A single BM25 retrieval result."""

    chunk: Chunk
    score: float
    term_count: int


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]


def bm25_retrieve(chunks: List[Chunk], query: str, k: int = 5) -> List[BM25Result]:
    """Retrieve chunks using a simple BM25 scoring implementation."""
    if not chunks:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return [BM25Result(chunk=chunk, score=0.0, term_count=0) for chunk in chunks[:k]]

    tokenized_docs = [_tokenize(chunk.text) for chunk in chunks]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avgdl = sum(doc_lengths) / max(1, len(doc_lengths))

    doc_freqs = Counter()
    for tokens in tokenized_docs:
        doc_freqs.update(set(tokens))

    results: List[BM25Result] = []
    for chunk, tokens in zip(chunks, tokenized_docs):
        freq_counter = Counter(tokens)
        score = 0.0
        for term in set(query_terms):
            if term not in freq_counter:
                continue
            tf = freq_counter[term]
            df = doc_freqs[term]
            idf = math.log((len(chunks) - df + 0.5) / (df + 0.5) + 1.0)
            numerator = tf * (1.5 + 1.0)
            denominator = tf + 1.5 * (1.0 - 0.75 + 0.75 * (len(tokens) / avgdl))
            score += idf * numerator / denominator

        results.append(BM25Result(chunk=chunk, score=score, term_count=len(tokens)))

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:k]
