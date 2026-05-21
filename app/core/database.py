"""
SQLAlchemy async models for multi-tenant data:
  Tenant → User → Document → QueryLog → WebhookEndpoint
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Boolean, DateTime, Integer,
    Float, Text, ForeignKey, Index, JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import get_settings

settings = get_settings()

engine        = create_async_engine(settings.sqlite_url, echo=False)
AsyncSession_ = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ── Tenant ─────────────────────────────────────────────────────

class Tenant(Base):
    __tablename__ = "tenants"

    id          = Column(String,  primary_key=True)   # slug: "acme-corp"
    name        = Column(String,  nullable=False)
    plan        = Column(String,  default="starter")  # starter|growth|enterprise
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    settings    = Column(JSON,    default=dict)       # per-tenant overrides

    users       = relationship("User",            back_populates="tenant")
    documents   = relationship("Document",        back_populates="tenant")
    query_logs  = relationship("QueryLog",        back_populates="tenant")
    webhooks    = relationship("WebhookEndpoint", back_populates="tenant")


# ── User ───────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id          = Column(String,  primary_key=True)
    tenant_id   = Column(String,  ForeignKey("tenants.id"), nullable=False)
    email       = Column(String,  nullable=False)
    hashed_pw   = Column(String,  nullable=False)
    role        = Column(String,  default="member")  # admin|member|viewer
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    tenant      = relationship("Tenant", back_populates="users")

    __table_args__ = (
        Index("ix_users_tenant_email", "tenant_id", "email", unique=True),
    )


# ── Document ───────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id          = Column(String,  primary_key=True)
    tenant_id   = Column(String,  ForeignKey("tenants.id"), nullable=False)
    source_type = Column(String,  nullable=False)  # github|confluence|pagerduty|upload
    source_ref  = Column(String,  nullable=False)  # repo URL, page ID, etc.
    title       = Column(String)
    doc_type    = Column(String)   # code|runbook|incident|api_doc|sop
    chunk_count = Column(Integer, default=0)
    last_indexed = Column(DateTime)
    metadata_   = Column("metadata", JSON, default=dict)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    tenant      = relationship("Tenant", back_populates="documents")

    __table_args__ = (
        Index("ix_docs_tenant_source", "tenant_id", "source_ref"),
    )


# ── Query Log ──────────────────────────────────────────────────

class QueryLog(Base):
    __tablename__ = "query_logs"

    id              = Column(String,  primary_key=True)
    tenant_id       = Column(String,  ForeignKey("tenants.id"), nullable=False)
    user_id         = Column(String,  ForeignKey("users.id"),   nullable=True)
    question        = Column(Text,    nullable=False)
    query_type      = Column(String)
    answer          = Column(Text)
    sources         = Column(JSON,    default=list)
    iterations      = Column(Integer, default=1)
    latency_ms      = Column(Float)
    confidence      = Column(Float)
    # RAGAS scores (async — populated after response)
    faithfulness    = Column(Float)
    answer_relevancy = Column(Float)
    context_precision = Column(Float)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    tenant          = relationship("Tenant", back_populates="query_logs")

    __table_args__ = (
        Index("ix_qlogs_tenant_created", "tenant_id", "created_at"),
    )


# ── Webhook Endpoint ───────────────────────────────────────────

class WebhookEndpoint(Base):
    __tablename__ = "webhook_endpoints"

    id          = Column(String,  primary_key=True)
    tenant_id   = Column(String,  ForeignKey("tenants.id"), nullable=False)
    url         = Column(String,  nullable=False)
    secret      = Column(String,  nullable=False)   # per-endpoint HMAC secret
    events      = Column(JSON,    default=list)     # ["query.completed", "index.done"]
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    tenant      = relationship("Tenant", back_populates="webhooks")


# ── DB lifecycle ───────────────────────────────────────────────

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    async with AsyncSession_() as session:
        yield session
