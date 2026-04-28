"""
Centralised configuration using pydantic-settings.
Loads from .env and environment variables.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env and environment."""

    # ── Google Drive ─────────────────────────────────────────────────────
    google_auth_method: str = "oauth"  # "oauth" or "service_account"
    google_oauth_client_secret_file: Path = Path("credentials/oauth_client_secret.json")
    google_service_account_file: Path = Path("credentials/service_account.json")
    google_token_file: Path = Path(".cache/google_token.json")
    gdrive_folder_id_list: Optional[list[str]] = None  # Specific folders to sync

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 512  # tokens per chunk
    chunk_overlap: int = 64  # tokens of overlap between chunks

    # ── Search ───────────────────────────────────────────────────────────
    top_k: int = 5  # default number of chunks to retrieve per query
    faiss_index_path: Path = Path(".cache/faiss_index")
    metadata_path: Path = Path(".cache/metadata.json")

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    openai_api_key: str = ""
    openai_base_url: Optional[str] = None  # For Azure OpenAI, Ollama, etc.
    use_mock_llm: bool = False  # Set to True for testing without OpenAI API

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
