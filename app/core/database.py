"""
SQLAlchemy async models for multi-tenant data:
  Tenant → User → Document → QueryLog → WebhookEndpoint
"""

from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.core.config import get_settings
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy import (
    Column, String, Boolean, DateTime, Integer,
    Float, Text, ForeignKey, Index, JSON, null,
)


settings = get_settings()


engine        = create_async_engine(settings.sqlite_url, echo=False)
AsyncSession_ = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass
# ── Tenant ─────────────────────────────────────────────────────

class Tenant(Base):
    __tablename__ = "tenants"
    
    id = Column(String, primary_key=True)   # slug: "acme-corp"
    name = Column(String, nullable=False)
    plan = Column(String, default="starter")    # starter|growth|enterprise
    is_active = Column(Boolean, nullable=False)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    settings    = Column(JSON,    default=dict)       # per-tenant overrides

    users       = relationship("User",            back_populates="tenant")
    documents   = relationship("Document",        back_populates="tenant")
    query_logs  = relationship("QueryLog",        back_populates="tenant")
    webhooks    = relationship("WebhookEndpoint", back_populates="tenant")


# ── User ───────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key = True)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    email = Column(String, nullable=False)
    hashed_pw = Column(String, nullable=False)
    role = Column(String, default="member")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.UTC))

    tenant = relationship("Tenant", back_populates="users")

    __table_args__ = (
        Index("ix_users_tenant_email", "tenant_id", "email", unique=True)
    )


async def get_db() -> AsyncSession:
    async with AsyncSession_() as session:
        yield session