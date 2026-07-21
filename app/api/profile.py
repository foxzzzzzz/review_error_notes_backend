from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_student
from app.database import get_db
from app.models.student import Student
from app.schemas.profile import ProfileOut, ProfileUpdate


router = APIRouter(prefix="/profile", tags=["profile"])


def _profile_out(student: Student) -> ProfileOut:
    return ProfileOut(
        nickname=student.nickname,
        avatar_url=student.avatar_url,
        grade=student.grade,
        semester=student.semester,
        phone_bound=student.phone_bound,
        phone_masked="****" if student.phone_bound else "",
    )


@router.get("", response_model=ProfileOut)
async def get_profile(
    student_id: str = Depends(get_current_student),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Student).where(Student.id == student_id))
    return _profile_out(result.scalar_one())


@router.patch("", response_model=ProfileOut)
async def update_profile(
    data: ProfileUpdate,
    student_id: str = Depends(get_current_student),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Student).where(Student.id == student_id))
    student = result.scalar_one()
    for key, value in data.model_dump(exclude_unset=True, exclude_none=True).items():
        setattr(student, key, value)
    await db.commit()
    await db.refresh(student)
    return _profile_out(student)
