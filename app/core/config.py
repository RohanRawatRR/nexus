from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # ── App ────────────────────────────────────────────────────
    app_name:           str  = "Nexus Engineering Copilot"
    app_version:        str  = "1.0.0"
    environment:        str  = "development"   # development | staging | production
    debug:              bool = False

    # ── Security ───────────────────────────────────────────────
    secret_key:         str  = "change-me-in-production-32-chars-min"
    jwt_algorithm:      str  = "HS256"
    jwt_expire_minutes: int  = 60 * 24        # 24 hours

    # ── OpenAI ────────────────────────────────────────────────
    openai_api_key:     str
    embedding_model:    str  = "text-embedding-3-large"
    chat_model:         str  = "gpt-4o"
    reranker_model:     str  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Storage ────────────────────────────────────────────────
    chroma_base_dir:    str  = "./data/chroma"    # tenant DBs: {base}/{tenant_id}/
    bm25_base_dir:      str  = "./data/bm25"
    sqlite_url:         str  = "sqlite+aiosqlite:///./data/nexus.db"

    # ── Redis ─────────────────────────────────────────────────
    redis_url:          str  = "redis://localhost:6379/0"
    cache_ttl_seconds:  int  = 300               # 5 min query cache

    # ── Rate limiting ─────────────────────────────────────────
    rate_limit_per_minute: int = 60              # per tenant
    rate_limit_per_hour:   int = 1000

    # ── RAG ───────────────────────────────────────────────────
    chunk_size_code:    int  = 150               # tokens — smaller for code
    chunk_size_docs:    int  = 300               # tokens — prose documents
    chunk_overlap:      int  = 30
    parent_chunk_size:  int  = 600
    top_k_retrieval:    int  = 20
    top_k_rerank:       int  = 6
    max_rag_iterations: int  = 3
    adequacy_threshold: float = 0.65

    # ── Integrations ──────────────────────────────────────────
    slack_bot_token:    Optional[str] = None
    slack_signing_secret: Optional[str] = None
    teams_app_id:       Optional[str] = None
    teams_app_password: Optional[str] = None
    webhook_secret:     str  = "change-me-webhook-secret"

    # ── Analytics ─────────────────────────────────────────────
    analytics_retention_days: int = 90

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
