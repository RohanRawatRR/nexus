"""
Auth layer:
  - JWT tokens carry tenant_id + user_id + role
  - Every protected endpoint gets a CurrentUser dependency
  - RBAC: require_role("admin") decorator for admin-only routes
  - Tenant context injected into every request via middleware
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
from functools import wraps

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config   import get_settings
from app.core.database import get_db, User, Tenant

settings    = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer      = HTTPBearer()


# ── Schemas ────────────────────────────────────────────────────

class TokenPayload(BaseModel):
    sub:       str         # user_id
    tenant_id: str
    role:      str
    exp:       datetime


class CurrentUser(BaseModel):
    user_id:   str
    tenant_id: str
    role:      str
    email:     str

    class Config:
        from_attributes = True


# ── Token utils ────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, tenant_id: str, role: str) -> str:
    expire  = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": user_id, "tenant_id": tenant_id, "role": role, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> TokenPayload:
    try:
        data = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        return TokenPayload(**data)
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")


# ── FastAPI dependencies ───────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db:          AsyncSession                 = Depends(get_db),
) -> CurrentUser:
    payload = decode_token(credentials.credentials)

    user = (await db.execute(
        select(User).where(User.id == payload.sub, User.is_active == True)
    )).scalar_one_or_none()

    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")

    # Validate tenant is still active
    tenant = (await db.execute(
        select(Tenant).where(Tenant.id == user.tenant_id, Tenant.is_active == True)
    )).scalar_one_or_none()

    if not tenant:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Tenant suspended")

    return CurrentUser(
        user_id   = user.id,
        tenant_id = user.tenant_id,
        role      = user.role,
        email     = user.email,
    )


def require_role(*roles: str):
    """
    Dependency factory for role-based access control.
    Usage: Depends(require_role("admin", "member"))
    """
    async def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role '{user.role}' cannot access this resource. Required: {roles}",
            )
        return user
    return _check


# Convenience aliases
AdminUser  = Depends(require_role("admin"))
MemberUser = Depends(require_role("admin", "member"))
AnyUser    = Depends(get_current_user)
