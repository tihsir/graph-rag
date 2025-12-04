"""Configuration management using Pydantic settings."""

from __future__ import annotations  # Enable postponed evaluation for Python 3.9 compatibility

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Provider
    llm_provider: Literal["openai", "anthropic", "vllm"] = "openai"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    vllm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "gpt-3.5-turbo"  # Higher rate limits (TPM/RPM) than gpt-4o-mini. Alternatives: gpt-4o-mini, gpt-4o, gpt-4-turbo

    # Embeddings Provider
    embedding_provider: Literal["openai", "local"] = "local"  # Default to local to avoid API quota issues
    openai_embedding_model: str = "text-embedding-3-large"
    local_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"  # Better quality than MiniLM

    # Reranker Provider
    reranker_provider: Literal["local", "cohere"] = "local"
    cohere_api_key: Optional[str] = None
    local_reranker_model: str = "BAAI/bge-reranker-base"

    # Vector Store
    vector_store: Literal["faiss", "qdrant"] = "faiss"
    qdrant_url: str = "http://localhost:6333"
    faiss_index_path: str = "./data/faiss_index"

    # Graph Store
    graph_store: Literal["nx_pg", "neo4j"] = "nx_pg"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Database
    database_url: str = "sqlite:///./data/graphrag.db"  # Default to SQLite for easy local testing
    metadata_db_path: str = "./data/metadata.db"  # SQLite for metadata store
    redis_url: str = "redis://localhost:6379/0"
    use_cache: bool = False

    # Processing Settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_context_chunks: int = 20
    max_graph_nodes: int = 100
    max_edges: int = 200
    min_confidence: float = 0.4
    max_hops: int = 2
    batch_size: int = 64

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Tracing
    enable_tracing: bool = True
    trace_storage_path: str = "./data/traces"


settings = Settings()

