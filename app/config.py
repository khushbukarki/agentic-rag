"""Runtime configuration, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    # "sentence-transformers" in production, "hash" for fast tests / CI
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "sentence-transformers"))
    # "chroma" in production, "memory" for tests
    vector_store: str = field(default_factory=lambda: os.getenv("VECTOR_STORE", "chroma"))
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "./chroma_data"))
    sample_docs_dir: str = field(default_factory=lambda: os.getenv("SAMPLE_DOCS_DIR", "./data/sample_docs"))
    load_sample_docs: bool = field(
        default_factory=lambda: os.getenv("LOAD_SAMPLE_DOCS", "true").lower() == "true"
    )
    chunk_words: int = field(default_factory=lambda: int(os.getenv("CHUNK_WORDS", "120")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "30")))


def get_settings() -> Settings:
    return Settings()
