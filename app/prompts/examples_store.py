from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional

from openai import OpenAI, OpenAIError

from app.config import get_settings
from app.prompts.faq_examples import get_faq_examples


@dataclass
class Example:
    user: str
    assistant: str
    intent: str
    category: str
    embedding: Optional[List[float]] = None


_cosine_similarity_cache: Dict[str, float] = {}


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    """Compute cosine similarity between two vectors with memoization on tuple keys."""
    key = (tuple(round(v, 6) for v in left), tuple(round(v, 6) for v in right))
    cached = _cosine_similarity_cache.get(key)
    if cached is not None:
        return cached
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sqrt(sum(a * a for a in left))
    norm_right = sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    result = dot / (norm_left * norm_right)
    _cosine_similarity_cache[key] = result
    return result


def _build_examples() -> List[Example]:
    """Build a list of Example objects from the raw FAQ examples."""
    raw_examples = get_faq_examples()
    examples: List[Example] = []
    for item in raw_examples:
        examples.append(
            Example(
                user=item["user"],
                assistant=item["assistant"],
                intent=item.get("intent", "OTHER"),
                category=item.get("category", "general"),
            )
        )
    return examples


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using OpenAI embeddings. Returns zero vectors on failure."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your-"):
        return [[0.0] * 1536 for _ in texts]

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Precomputed embedding store (computed once at import time)
# ---------------------------------------------------------------------------

_EXAMPLES: List[Example] = []
_EXAMPLES_LOADED = False


def _load_examples_with_embeddings() -> List[Example]:
    """Build examples and precompute embeddings for the user texts."""
    global _EXAMPLES, _EXAMPLES_LOADED
    if _EXAMPLES_LOADED:
        return _EXAMPLES

    examples = _build_examples()
    try:
        user_texts = [example.user for example in examples]
        embeddings = _embed_texts(user_texts)
        for example, embedding in zip(examples, embeddings):
            example.embedding = embedding
    except (OpenAIError, TypeError, ValueError, IndexError, AttributeError):
        pass
    _EXAMPLES = examples
    _EXAMPLES_LOADED = True
    return _EXAMPLES


def select_examples(query: str, k: int = 3) -> List[Example]:
    """Select top-k examples based on cosine similarity between query and example user texts."""
    examples = _load_examples_with_embeddings()

    try:
        query_embedding = _embed_texts([query])[0]
        scored = []
        for example in examples:
            if example.embedding is None:
                continue
            similarity = _cosine_similarity(example.embedding, query_embedding)
            scored.append((example, similarity))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        selected = [example for example, _score in scored[:k]]
        if selected:
            return selected
    except (OpenAIError, TypeError, ValueError, IndexError, AttributeError):
        pass

    # Fallback: return first k examples if embedding fails
    return examples[:k]


def get_all_examples() -> List[Example]:
    """Return all cached examples (with embeddings, if computed)."""
    return _load_examples_with_embeddings()


def refresh_examples() -> None:
    """Force reload and re-embed examples."""
    global _EXAMPLES_LOADED
    _EXAMPLES_LOADED = False
    _load_examples_with_embeddings()