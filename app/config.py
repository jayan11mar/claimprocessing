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
    SQLITE_DB_PATH: str = Field("claims.db", env="SQLITE_DB_PATH")
    ENVIRONMENT: str = Field("dev", env="ENVIRONMENT")

    LANGSMITH_API_KEY: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    LANGSMITH_TRACING: Optional[bool] = Field(False, env="LANGSMITH_TRACING")
    LANGSMITH_PROJECT_NAME: Optional[str] = Field(None, env="LANGSMITH_PROJECT_NAME")

    # RAG settings
    OPENAI_EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    EMBEDDING_MODEL: Optional[str] = Field(None, env="EMBEDDING_MODEL")
    VECTOR_BACKEND: str = Field("faiss", env="VECTOR_BACKEND")
    PINECONE_API_KEY: Optional[str] = Field(None, env="PINECONE_API_KEY")
    KNOWLEDGE_BASE_DIR: Optional[str] = Field(None, env="KNOWLEDGE_BASE_DIR")
    CHUNK_SIZE: int = Field(800, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(100, env="CHUNK_OVERLAP")
    CHUNKING_STRATEGY: str = Field("recursive", env="CHUNKING_STRATEGY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
