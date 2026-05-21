from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ────────────────────────────────────────────────────
    app_name:           str  = "Nexus Engineering Copilot"
    app_version:        str  = "1.0.0"
    environment:        str  = "development"   # development | staging | production
    debug:              bool = False

    # ── Storage ────────────────────────────────────────────────
    chroma_base_dir:    str  = "./data/chroma"    # tenant DBs: {base}/{tenant_id}/
    bm25_base_dir:      str  = "./data/bm25"
    sqlite_url:         str  = "sqlite+aiosqlite:///./data/nexus.db"

@lru_cache
def get_settings() -> Settings:
    return Settings()
