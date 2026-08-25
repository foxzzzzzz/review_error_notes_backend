from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import func, select

from app.models.wrong_question import WrongQuestion
from app.schemas.profile import ProfileStats, ProfileUpdate


BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def beijing_month_start_utc_naive(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current_beijing = current.astimezone(BEIJING_TIMEZONE)
    month_start_beijing = current_beijing.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return month_start_beijing.astimezone(timezone.utc).replace(tzinfo=None)


async def load_profile_stats(
    db,
    student_id,
    now: datetime | None = None,
) -> ProfileStats:
    month_start = beijing_month_start_utc_naive(now)
    statement = select(
        func.count(WrongQuestion.id),
        func.count(WrongQuestion.id).filter(
            WrongQuestion.created_at >= month_start
        ),
        func.count(WrongQuestion.id).filter(
            WrongQuestion.mastery_status == "learning"
        ),
        func.count(WrongQuestion.id).filter(
            WrongQuestion.mastery_status == "mastered"
        ),
    ).where(
        WrongQuestion.student_id == student_id,
        WrongQuestion.deleted_at.is_(None),
        WrongQuestion.collection_status == "collected",
    )
    row = (await db.execute(statement)).one()
    return ProfileStats(
        total=row[0] or 0,
        month_new=row[1] or 0,
        learning=row[2] or 0,
        mastered=row[3] or 0,
    )


def apply_profile_update(
    account,
    student,
    data: ProfileUpdate,
    now: datetime | None = None,
) -> None:
    updated_at = now or _utc_now_naive()
    changes = data.model_dump(exclude_unset=True, exclude_none=True)

    if "nickname" in changes:
        account.nickname = changes["nickname"]
        account.profile_prompted_at = account.profile_prompted_at or updated_at
        account.profile_completed_at = updated_at
    if "student_name" in changes:
        student.display_name = changes["student_name"]
    if "grade" in changes:
        student.grade = changes["grade"]
    if "semester" in changes:
        student.semester = changes["semester"]

    student.profile_completed = (
        student.grade is not None and student.semester is not None
    )


def mark_profile_prompt_skipped(
    account,
    now: datetime | None = None,
) -> None:
    account.profile_prompted_at = (
        account.profile_prompted_at or now or _utc_now_naive()
    )
