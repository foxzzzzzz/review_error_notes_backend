from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_default_student
from app.database import get_db
from app.models.account import Account
from app.models.student import Student
from app.schemas.profile import ProfileOut, ProfileUpdate


router = APIRouter(prefix="/profile", tags=["profile"])


def _profile_out(account: Account, student: Student) -> ProfileOut:
    return ProfileOut(
        nickname=account.nickname,
        avatar_url=account.avatar_object_key,
        grade=student.grade,
        semester=student.semester,
        phone_bound=account.phone_ciphertext is not None,
        phone_masked="****" if account.phone_ciphertext else "",
    )


@router.get("", response_model=ProfileOut)
async def get_profile(
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    account = await db.scalar(select(Account).where(Account.id == student.account_id))
    return _profile_out(account, student)


@router.patch("", response_model=ProfileOut)
async def update_profile(
    data: ProfileUpdate,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    for key, value in data.model_dump(exclude_unset=True, exclude_none=True).items():
        setattr(student, key, value)
    await db.commit()
    await db.refresh(student)
    account = await db.scalar(select(Account).where(Account.id == student.account_id))
    return _profile_out(account, student)
