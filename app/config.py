"""Application configuration, read from environment variables.

Every knob lives here. All settings are prefixed with DOCCHAT_ so nothing
collides with other services. See `.env.example` for the full list.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCCHAT_", env_file=".env", extra="ignore"
    )

    # --- core ---------------------------------------------------------------
    app_name: str = "docchat"
    version: str = "2.0.0"
    data_dir: str = "data"
    db_path: str = "data/docchat.db"
    # Full database URL. When set, it wins over db_path. Supports SQLite
    # (sqlite:///path) and Postgres (postgresql://user:pass@host/db), so the
    # same code runs locally on SQLite and in production on Postgres.
    database_url: str = ""

    # --- auth ---------------------------------------------------------------
    # Set DOCCHAT_AUTH_ENABLED=false for a quick local demo without logins.
    # When enabled, register/login return a JWT used as a bearer token.
    auth_enabled: bool = True
    secret_key: str = "dev-secret-change-me"
    token_expiry_minutes: int = 60 * 24 * 7  # 7 days
    demo_user_email: str = "demo@docchat.local"

    # --- embeddings ---------------------------------------------------------
    # Provider: "tfidf" (offline, default) | "openai" | "local"
    # "local" uses sentence-transformers (pip install -r requirements-optional.txt)
    embedding_provider: str = "tfidf"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # --- generation (LLM) ---------------------------------------------------
    # Provider: "none" (retrieval-only) | "ollama" (local) | "openai" (compatible)
    llm_provider: str = "none"
    llm_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.2

    # --- retrieval ----------------------------------------------------------
    default_k: int = 4
    words_per_chunk: int = 120
    chunk_overlap: int = 30

    # --- docs / indexing ----------------------------------------------------
    docs_dir: str = "sample_docs"
    seed_demo_user: bool = True  # create demo user + index docs on startup

    # --- web ----------------------------------------------------------------
    # Comma-separated list of allowed CORS origins ("" disables the check).
    allowed_origins: str = "*"
    # Built frontend is served from here if the directory exists.
    frontend_dist: str = "frontend/dist"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
