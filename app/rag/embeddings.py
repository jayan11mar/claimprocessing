"""
Embedding functions for the claims knowledge base.
Wraps OpenAI embeddings and sentence-transformer models.
"""

from typing import Any, Callable, Dict, List, Optional

from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from app.config import get_settings


def get_embedding_fn(model_name: Optional[str] = None) -> Callable[[List[str]], List[List[float]]]:
    """
    Get an embedding function based on the model name.
    Returns a callable that can be used by vector stores.

    Args:
        model_name: Name of the embedding model.
                   - OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
                   - Sentence-transformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2", etc.
                   If None, uses OPENAI_EMBEDDING_MODEL from config or defaults to "text-embedding-3-small".

    Returns:
        A callable that takes a list of strings and returns a list of embeddings.
    """
    settings = get_settings()

    if model_name is None:
        model_name = getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Check if it's an OpenAI model
    openai_models = {
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    }

    if model_name in openai_models or model_name.startswith("text-embedding"):
        return _get_openai_embedding_fn(model_name)
    else:
        return _get_sentence_transformer_embedding_fn(model_name)


def _get_openai_embedding_fn(model_name: str) -> Callable[[List[str]], List[List[float]]]:
    """
    Get OpenAI embedding function.

    Args:
        model_name: OpenAI embedding model name.

    Returns:
        Embedding function for OpenAI models.
    """
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    def embed_fn(texts: List[str]) -> List[List[float]]:
        return embeddings.embed_documents(texts)

    return embed_fn


def _get_sentence_transformer_embedding_fn(model_name: str) -> Callable[[List[str]], List[List[float]]]:
    """
    Get sentence-transformer embedding function.

    Args:
        model_name: Sentence-transformer model name.

    Returns:
        Embedding function for sentence-transformer models.
    """
    model = SentenceTransformer(model_name)

    def embed_fn(texts: List[str]) -> List[List[float]]:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    return embed_fn


class EmbeddingCache:
    """Simple cache for embedding results."""

    def __init__(self, embedding_fn: Callable[[List[str]], List[List[float]]]):
        self._embedding_fn = embedding_fn
        self._cache: Dict[str, List[float]] = {}

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with caching."""
        results = []
        to_embed = []

        for text in texts:
            if text in self._cache:
                results.append(self._cache[text])
            else:
                to_embed.append(text)

        if to_embed:
            new_embeddings = self._embedding_fn(to_embed)
            for text, embedding in zip(to_embed, new_embeddings):
                self._cache[text] = embedding
            results.extend(new_embeddings)

        return results