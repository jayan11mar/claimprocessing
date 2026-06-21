from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional

from openai import OpenAI

from app.config import get_settings
from app.prompts.faq_examples import get_faq_examples


@dataclass
class Example:
    user: str
    assistant: str
    intent: str
    category: str
    embedding: Optional[List[float]] = None


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sqrt(sum(a * a for a in left))
    norm_right = sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def _build_examples() -> List[Example]:
    raw_examples = get_faq_examples()
    return [
        Example(
            user=item["user"],
            assistant=item["assistant"],
            intent=item.get("intent", "OTHER"),
            category=item.get("category", "general"),
        )
        for item in raw_examples
    ]


def _embed_texts(texts: List[str]) -> List[List[float]]:
    settings = get_settings()
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your-"):
        return [[0.0] * 1536 for _ in texts]

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def select_examples(query: str, k: int = 3) -> List[Example]:
    """Select the top-k examples semantically most similar to the query."""
    examples = _build_examples()
    try:
        user_texts = [example.user for example in examples]
        embeddings = _embed_texts(user_texts)
        for example, embedding in zip(examples, embeddings):
            example.embedding = embedding

        query_embedding = _embed_texts([query])[0]
        scored = [
            (example, _cosine_similarity(example.embedding or [0.0], query_embedding))
            for example in examples
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        selected = [example for example, _score in scored[:k]]
        if selected:
            return selected
    except Exception:
        pass

    return examples[:k]
