from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_default_student
from app.config import settings
from app.database import get_db
from app.models.account import Account
from app.models.student import Student
from app.schemas.profile import ProfileOut, ProfileStats, ProfileUpdate
from app.services.profile import (
    apply_profile_update,
    load_profile_stats,
    mark_profile_prompt_skipped,
)
from app.services.avatar_storage import (
    AvatarInvalid,
    AvatarTooLarge,
    save_avatar_image,
)


router = APIRouter(prefix="/profile", tags=["profile"])


def _profile_out(
    account: Account,
    student: Student,
    stats: ProfileStats,
) -> ProfileOut:
    return ProfileOut(
        nickname=account.nickname,
        avatar_url=account.avatar_object_key,
        profile_prompt_required=(
            account.profile_prompted_at is None
            and account.profile_completed_at is None
        ),
        student_id=student.id,
        student_name=student.display_name,
        grade=student.grade,
        semester=student.semester,
        student_profile_required=(
            student.grade is None or student.semester is None
        ),
        phone_bound=account.phone_ciphertext is not None,
        phone_masked="****" if account.phone_ciphertext else "",
        stats=stats,
    )


@router.get("", response_model=ProfileOut)
async def get_profile(
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    account = await db.scalar(select(Account).where(Account.id == student.account_id))
    stats = await load_profile_stats(db, student.id)
    return _profile_out(account, student, stats)


@router.patch("", response_model=ProfileOut)
async def update_profile(
    data: ProfileUpdate,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    account = await db.scalar(
        select(Account)
        .where(Account.id == student.account_id)
        .with_for_update()
    )
    student = await db.scalar(
        select(Student)
        .where(
            Student.id == student.id,
            Student.account_id == account.id,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    apply_profile_update(account, student, data)
    await db.commit()
    await db.refresh(student)
    await db.refresh(account)
    stats = await load_profile_stats(db, student.id)
    return _profile_out(account, student, stats)


@router.post("/prompt/skip", response_model=ProfileOut)
async def skip_profile_prompt(
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    account = await db.scalar(
        select(Account)
        .where(Account.id == student.account_id)
        .with_for_update()
    )
    mark_profile_prompt_skipped(account)
    await db.commit()
    await db.refresh(account)
    stats = await load_profile_stats(db, student.id)
    return _profile_out(account, student, stats)


@router.post("/avatar", response_model=ProfileOut)
async def upload_avatar(
    file: UploadFile = File(...),
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    data = await file.read(settings.AVATAR_MAX_BYTES + 1)
    try:
        saved = save_avatar_image(
            data,
            avatar_dir=settings.AVATAR_DIR,
            max_bytes=settings.AVATAR_MAX_BYTES,
            max_edge=settings.AVATAR_MAX_EDGE,
            jpeg_quality=settings.AVATAR_JPEG_QUALITY,
        )
    except AvatarTooLarge as exc:
        raise HTTPException(status_code=413, detail="头像文件过大") from exc
    except AvatarInvalid as exc:
        raise HTTPException(status_code=422, detail="头像图片无效") from exc

    try:
        account = await db.scalar(
            select(Account)
            .where(Account.id == student.account_id)
            .with_for_update()
        )
        old_avatar_url = account.avatar_object_key
        account.avatar_object_key = saved.public_url
        await db.commit()
        await db.refresh(account)
    except Exception:
        await db.rollback()
        saved.path.unlink(missing_ok=True)
        raise

    if old_avatar_url and old_avatar_url.startswith("/avatars/"):
        old_path = Path(settings.AVATAR_DIR) / Path(old_avatar_url).name
        if old_path != saved.path:
            old_path.unlink(missing_ok=True)

    stats = await load_profile_stats(db, student.id)
    return _profile_out(account, student, stats)
