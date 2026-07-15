from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    OPENAI_MODEL_NAME: str = Field("gpt-4o-mini", env="OPENAI_MODEL_NAME")
    OPENAI_MODEL_TEMPERATURE: float = Field(0.8, env="OPENAI_MODEL_TEMPERATURE")
    OPENAI_REQUEST_TIMEOUT: float = Field(30.0, env="OPENAI_REQUEST_TIMEOUT")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    SQLITE_DB_PATH: str = Field("data/claims.db", env="SQLITE_DB_PATH")
    ENVIRONMENT: str = Field("dev", env="ENVIRONMENT")

    LANGSMITH_API_KEY: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    LANGSMITH_TRACING: Optional[bool] = Field(False, env="LANGSMITH_TRACING")
    LANGSMITH_PROJECT_NAME: Optional[str] = Field(None, env="LANGSMITH_PROJECT_NAME")

    # RAG settings
    OPENAI_EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    EMBEDDING_MODEL: Optional[str] = Field(None, env="EMBEDDING_MODEL")
    EMBEDDING_MODEL_VERSION: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL_VERSION")
    VECTOR_BACKEND: str = Field("faiss", env="VECTOR_BACKEND")
    PINECONE_API_KEY: Optional[str] = Field(None, env="PINECONE_API_KEY")
    KNOWLEDGE_BASE_DIR: Optional[str] = Field(None, env="KNOWLEDGE_BASE_DIR")
    VECTOR_PERSIST_PATH: str = Field("data/faiss_index", env="VECTOR_PERSIST_PATH")
    CHUNK_SIZE: int = Field(800, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(100, env="CHUNK_OVERLAP")
    CHUNKING_STRATEGY: str = Field("recursive", env="CHUNKING_STRATEGY")
    RAG_EVALUATION_CONTEXT: str = Field("claims / insurance", env="RAG_EVALUATION_CONTEXT")
    RAG_EVALUATION_RETRIEVAL_THRESHOLD: float = Field(0.85, env="RAG_EVALUATION_RETRIEVAL_THRESHOLD")
    RAG_EVALUATION_ANSWER_THRESHOLD: float = Field(0.8, env="RAG_EVALUATION_ANSWER_THRESHOLD")
    RAG_EVALUATION_OVERALL_THRESHOLD: float = Field(0.8, env="RAG_EVALUATION_OVERALL_THRESHOLD")
    RAG_EVALUATION_HIT_RATE_THRESHOLD: float = Field(0.85, env="RAG_EVALUATION_HIT_RATE_THRESHOLD")
    RAG_EVALUATION_MRR_THRESHOLD: float = Field(0.65, env="RAG_EVALUATION_MRR_THRESHOLD")
    RAG_EVALUATION_FAITHFULNESS_THRESHOLD: float = Field(0.9, env="RAG_EVALUATION_FAITHFULNESS_THRESHOLD")
    RAG_EVALUATION_ANSWER_CORRECTNESS_THRESHOLD: float = Field(0.8, env="RAG_EVALUATION_ANSWER_CORRECTNESS_THRESHOLD")
    RAG_EVALUATION_LLM_JUDGE_AVG_THRESHOLD: float = Field(4.0, env="RAG_EVALUATION_LLM_JUDGE_AVG_THRESHOLD")
    RAG_EVALUATION_CITATION_COVERAGE_THRESHOLD: float = Field(1.0, env="RAG_EVALUATION_CITATION_COVERAGE_THRESHOLD")
    RAG_EVALUATION_MIN_CITATIONS: int = Field(1, env="RAG_EVALUATION_MIN_CITATIONS")
    RAG_EVALUATION_OUTPUT_PATH: Optional[str] = Field(None, env="RAG_EVALUATION_OUTPUT_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
