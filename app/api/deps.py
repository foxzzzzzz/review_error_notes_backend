from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.account import Account
from app.models.student import Student
from app.utils.jwt import verify_token


security = HTTPBearer()


@dataclass(frozen=True)
class AccountContext:
    account_id: str
    token_version: int


async def get_current_account(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> AccountContext:
    claims = verify_token(credentials.credentials)
    if claims is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    result = await db.execute(
        select(Account).where(Account.id == claims.account_id)
    )
    account = result.scalar_one_or_none()
    if account is None or account.token_version != claims.token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    if account.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "account_pending_deletion",
                "message": "账户正在注销流程中",
            },
        )
    return AccountContext(
        account_id=str(account.id),
        token_version=account.token_version,
    )


async def get_default_student(
    context: AccountContext = Depends(get_current_account),
    db: AsyncSession = Depends(get_db),
) -> Student:
    result = await db.execute(
        select(Student).where(
            Student.account_id == context.account_id,
            Student.is_default.is_(True),
        )
    )
    student = result.scalar_one_or_none()
    if student is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "student_profile_required",
                "message": "请先完善当前学生资料",
            },
        )
    return student
