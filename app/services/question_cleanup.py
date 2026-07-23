"""Physical cleanup for expired soft-deleted questions and orphan images."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion


def cleanup_expired_questions(
    db: Session,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, int]:
    """Delete all expired questions and orphan images in locked batches."""
    cleanup_time = now or datetime.now(timezone.utc)
    if cleanup_time.tzinfo is not None:
        cleanup_time = cleanup_time.astimezone(timezone.utc).replace(tzinfo=None)
    cutoff = cleanup_time - timedelta(
        days=settings.QUESTION_SOFT_DELETE_RETENTION_DAYS
    )

    questions_deleted = 0
    while True:
        questions = db.scalars(
            select(WrongQuestion)
            .where(
                WrongQuestion.deleted_at.is_not(None),
                WrongQuestion.deleted_at < cutoff,
            )
            .order_by(WrongQuestion.deleted_at, WrongQuestion.id)
            .limit(settings.QUESTION_CLEANUP_BATCH_SIZE)
            .with_for_update(skip_locked=True)
        ).all()
        if not questions:
            break
        for question in questions:
            db.delete(question)
        db.commit()
        questions_deleted += len(questions)

    failed_image_ids = set()
    images_deleted = 0
    while True:
        filters = [
            WrongImage.created_at < cutoff,
            ~exists().where(WrongQuestion.image_id == WrongImage.id),
        ]
        if failed_image_ids:
            filters.append(WrongImage.id.not_in(failed_image_ids))
        images = db.scalars(
            select(WrongImage)
            .where(*filters)
            .order_by(WrongImage.created_at, WrongImage.id)
            .limit(settings.QUESTION_CLEANUP_BATCH_SIZE)
            .with_for_update(skip_locked=True)
        ).all()
        if not images:
            break

        batch_deleted = 0
        for image in images:
            filepath = Path(settings.UPLOAD_DIR) / Path(image.original_url).name
            try:
                filepath.unlink(missing_ok=True)
            except OSError:
                failed_image_ids.add(image.id)
                continue
            db.delete(image)
            batch_deleted += 1
        db.commit()
        images_deleted += batch_deleted

    return {
        "questions_deleted": questions_deleted,
        "images_deleted": images_deleted,
    }
