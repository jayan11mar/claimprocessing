"""
Embedding functions for the claims knowledge base.
Wraps OpenAI embeddings and sentence-transformer models.
"""

from typing import Any, Callable, Dict, List, Optional

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover - optional dependency guard
    OpenAIEmbeddings = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency guard
    SentenceTransformer = None

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
        if not getattr(settings, "OPENAI_API_KEY", None):
            return _get_fallback_embedding_fn(_get_openai_embedding_dimension(model_name))
        return _get_openai_embedding_fn(model_name)
    else:
        return _get_sentence_transformer_embedding_fn(model_name)


def get_embedding_model_version() -> str:
    """
    Return the version-pinned embedding model identifier.
    This is stored alongside the vector index to detect model drift
    between ingestion and query time.
    """
    settings = get_settings()
    return getattr(settings, "EMBEDDING_MODEL_VERSION", "text-embedding-3-small@2024-02-15")


def check_embedding_model_consistency(stored_version: Optional[str]) -> None:
    """
    Verify that the embedding model used at query time matches the one used at ingestion.
    Raises a warning if they differ.

    Args:
        stored_version: The embedding model version stored with the index at ingestion time.
    """
    if stored_version is None:
        return
    current_version = get_embedding_model_version()
    if current_version != stored_version:
        import warnings
        warnings.warn(
            f"Embedding model mismatch detected! "
            f"Index was built with '{stored_version}' but current config uses '{current_version}'. "
            f"Re-ingest the knowledge base to ensure consistent retrieval quality.",
            RuntimeWarning,
            stacklevel=2,
        )


def _get_openai_embedding_fn(model_name: str) -> Callable[[List[str]], List[List[float]]]:
    """
    Get OpenAI embedding function.

    Args:
        model_name: OpenAI embedding model name.

    Returns:
        Embedding function for OpenAI models.
    """
    if OpenAIEmbeddings is None:
        return _get_fallback_embedding_fn(_get_openai_embedding_dimension(model_name))

    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    def embed_fn(texts: List[str]) -> List[List[float]]:
        return embeddings.embed_documents(texts)

    return embed_fn


def _get_openai_embedding_dimension(model_name: str) -> int:
    """Get the embedding dimension for an OpenAI model.
    
    Args:
        model_name: OpenAI embedding model name.
        
    Returns:
        The embedding dimension for the model.
    """
    # OpenAI embedding model dimensions
    # text-embedding-3-small: 1536 (default)
    # text-embedding-3-large: 3072
    # text-embedding-ada-002: 1536
    if "text-embedding-3-large" in model_name:
        return 3072
    return 1536  # Default for text-embedding-3-small and text-embedding-ada-002


def _get_sentence_transformer_embedding_fn(model_name: str) -> Callable[[List[str]], List[List[float]]]:
    """
    Get sentence-transformer embedding function.

    Args:
        model_name: Sentence-transformer model name.

    Returns:
        Embedding function for sentence-transformer models.
    """
    if SentenceTransformer is None:
        return _get_fallback_embedding_fn(_get_sentence_transformer_dimension(model_name))

    model = SentenceTransformer(model_name)

    def embed_fn(texts: List[str]) -> List[List[float]]:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    return embed_fn


def _get_sentence_transformer_dimension(model_name: str) -> int:
    """Get the embedding dimension for a sentence-transformer model.
    
    Args:
        model_name: Sentence-transformer model name.
        
    Returns:
        The embedding dimension for the model.
    """
    # Common sentence-transformer model dimensions
    # all-MiniLM-L6-v2: 384
    # all-mpnet-base-v2: 768
    # all-MiniLM-L12-v2: 384
    # paraphrase-MiniLM-L6-v2: 384
    # paraphrase-mpnet-base-v2: 768
    # Default to 384 for most MiniLM models
    if "mpnet" in model_name.lower():
        return 768
    return 384  # Default for most MiniLM models


def _get_fallback_embedding_fn(dimension: int = 1536) -> Callable[[List[str]], List[List[float]]]:
    """Return a deterministic placeholder embedding function when heavyweight deps are unavailable.
    
    Args:
        dimension: The embedding dimension to return. Defaults to 1536 to match
                   text-embedding-3-small and the default Pinecone index dimension.
    
    Returns:
        Embedding function that returns vectors of the specified dimension.
    """

    def embed_fn(texts: List[str]) -> List[List[float]]:
        # Generate a deterministic embedding of the correct dimension
        return [[sum(ord(c) * (i + 1) for c in text) % 100 / 100.0 for i in range(dimension)] for text in texts]

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