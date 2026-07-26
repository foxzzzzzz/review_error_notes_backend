"""Physical cleanup for accounts whose recovery period has expired."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Optional

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.account import Account
from app.models.account_recovery import FileCleanupJob
from app.models.practice_attempt import PracticeAttempt
from app.models.practice_result import PracticeResult
from app.models.practice_sheet import PracticeSheet
from app.models.sheet_item import SheetItem
from app.models.student import Student
from app.models.wechat_identity import WeChatIdentity
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion
from app.services.file_cleanup import _delete_cleanup_path


@dataclass(frozen=True)
class AccountCleanupSummary:
    accounts_deleted: int = 0
    files_queued: int = 0
    files_deleted: int = 0
    file_failures: int = 0


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _cleanup_job(path: Optional[str]) -> Optional[FileCleanupJob]:
    if not path:
        return None
    normalized = PurePosixPath(path)
    if len(normalized.parts) != 3:
        return None
    storage_kind = {
        "/avatars": "avatar",
        "/uploads": "upload",
        "/pdfs": "pdf",
    }.get(f"/{normalized.parts[1]}")
    if storage_kind is None or normalized.name in {"", ".", ".."}:
        return None
    return FileCleanupJob(
        storage_kind=storage_kind,
        object_path=str(normalized),
    )


def _delete_account_data(
    db: Session,
    account: Account,
) -> list[FileCleanupJob]:
    student_ids = select(Student.id).where(Student.account_id == account.id)
    sheet_ids = select(PracticeSheet.id).where(
        PracticeSheet.student_id.in_(student_ids)
    )
    attempt_ids = select(PracticeAttempt.id).where(
        PracticeAttempt.student_id.in_(student_ids)
    )

    db.execute(
        delete(PracticeResult).where(
            PracticeResult.attempt_id.in_(attempt_ids)
        )
    )
    db.execute(
        delete(PracticeAttempt).where(
            PracticeAttempt.student_id.in_(student_ids)
        )
    )
    db.execute(delete(SheetItem).where(SheetItem.sheet_id.in_(sheet_ids)))
    pdf_rows = db.execute(
        delete(PracticeSheet)
        .where(PracticeSheet.student_id.in_(student_ids))
        .returning(PracticeSheet.pdf_url)
    ).all()
    db.execute(
        delete(WrongQuestion).where(
            WrongQuestion.student_id.in_(student_ids)
        )
    )
    upload_rows = db.execute(
        delete(WrongImage)
        .where(WrongImage.student_id.in_(student_ids))
        .returning(WrongImage.original_url)
    ).all()
    db.execute(delete(Student).where(Student.account_id == account.id))
    db.execute(
        delete(WeChatIdentity).where(
            WeChatIdentity.account_id == account.id
        )
    )
    db.execute(delete(Account).where(Account.id == account.id))

    paths = [
        account.avatar_object_key,
        *(row[0] for row in pdf_rows),
        *(row[0] for row in upload_rows),
    ]
    return [job for path in paths if (job := _cleanup_job(path)) is not None]


def _retry_file_cleanup_jobs(
    db: Session,
    *,
    cleanup_time: datetime,
) -> tuple[int, int]:
    deleted = 0
    failures = 0
    while True:
        jobs = db.scalars(
            select(FileCleanupJob)
            .where(
                or_(
                    FileCleanupJob.next_attempt_at.is_(None),
                    FileCleanupJob.next_attempt_at <= cleanup_time,
                )
            )
            .order_by(FileCleanupJob.created_at, FileCleanupJob.id)
            .limit(settings.ACCOUNT_CLEANUP_BATCH_SIZE)
            .with_for_update(skip_locked=True)
        ).all()
        if not jobs:
            break

        for job in jobs:
            try:
                _delete_cleanup_path(job)
            except (OSError, ValueError) as exc:
                job.attempt_count = (job.attempt_count or 0) + 1
                job.last_error = type(exc).__name__
                job.next_attempt_at = cleanup_time + timedelta(
                    seconds=settings.ACCOUNT_CLEANUP_INTERVAL_SECONDS
                )
                failures += 1
                continue
            db.delete(job)
            deleted += 1
        db.commit()
    return deleted, failures


def cleanup_expired_accounts(
    db: Session,
    *,
    now: Optional[datetime] = None,
) -> AccountCleanupSummary:
    cleanup_time = _naive_utc(now or datetime.now(timezone.utc))
    accounts_deleted = 0
    files_queued = 0
    while True:
        accounts = db.scalars(
            select(Account)
            .where(
                Account.status == "pending_deletion",
                Account.deletion_due_at.is_not(None),
                Account.deletion_due_at <= cleanup_time,
            )
            .order_by(Account.deletion_due_at, Account.id)
            .limit(settings.ACCOUNT_CLEANUP_BATCH_SIZE)
            .with_for_update(skip_locked=True)
        ).all()
        if not accounts:
            break

        queued_jobs: list[FileCleanupJob] = []
        for account in accounts:
            queued_jobs.extend(_delete_account_data(db, account))
        if queued_jobs:
            db.add_all(queued_jobs)
        db.commit()
        accounts_deleted += len(accounts)
        files_queued += len(queued_jobs)

    files_deleted, file_failures = _retry_file_cleanup_jobs(
        db,
        cleanup_time=cleanup_time,
    )
    return AccountCleanupSummary(
        accounts_deleted=accounts_deleted,
        files_queued=files_queued,
        files_deleted=files_deleted,
        file_failures=file_failures,
    )
