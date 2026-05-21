import select
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from core.schemas import TokenResponse, LoginRequest
from sqlalchemy.ext.asyncio import AsyncSession
from app.core. database import get_db
from app.core.database import User
from app.core.auth import verify_password

router = APIRouter()

@router.post('/auth/login', response_model=TokenResponse, tags=["auth"])
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(
        select(User).where(User.email == req.email, User.is_active == True)
    )).scalar_one_or_none()

    if not user or not verify_password(req.password, user.hashed_pw):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
