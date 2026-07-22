"""Celery task for MiniMax multimodal wrong-question recognition."""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import settings
# Register the complete SQLAlchemy foreign-key graph before sync-session flushes.
from app.models.practice_sheet import PracticeSheet  # noqa: F401
from app.models.sheet_item import SheetItem  # noqa: F401
from app.models.student import Student  # noqa: F401
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion
from app.services.vision_recognition import (
    MiniMaxVisionClient,
    build_question_values,
    image_status_for,
)
from app.tasks.celery_app import celery_app


@celery_app.task(bind=True)
def process_image(self, image_id: str, filepath: str):
    """Recognize an uploaded image with MiniMax and persist validated items."""
    sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(sync_url)

    from app.models import Base
    Base.metadata.create_all(engine, checkfirst=True)

    claimed = False
    try:
        with Session(engine) as db:
            image = db.scalar(
                select(WrongImage)
                .where(WrongImage.id == image_id)
                .with_for_update()
            )
            if not image or image.status != "pending":
                return
            subject_hint = image.subject
            image.status = "segmented"
            db.commit()
            claimed = True

        result = MiniMaxVisionClient.from_settings().recognize(
            filepath,
            subject_hint=subject_hint,
        )
        question_values = [
            build_question_values(
                item,
                index=index,
                confidence_threshold=settings.MINIMAX_CONFIDENCE_THRESHOLD,
            )
            for index, item in enumerate(result.items)
        ]

        with Session(engine) as db:
            image = db.scalar(
                select(WrongImage)
                .where(WrongImage.id == image_id)
                .with_for_update()
            )
            if not image or image.status != "segmented":
                return

            for values in question_values:
                values["ocr_raw_json"]["ignored_text"] = result.ignored_text
                question = WrongQuestion(
                    student_id=image.student_id,
                    image_id=image.id,
                    grade=image.grade,
                    semester=image.semester,
                    **values,
                )
                db.add(question)

            image.question_count = len(question_values)
            image.status = image_status_for(question_values)
            if not image.subject:
                image.subject = result.items[0].subject
            db.commit()
            claimed = False
    except Exception:
        if claimed:
            with Session(engine) as db:
                claimed_image = db.scalar(
                    select(WrongImage)
                    .where(WrongImage.id == image_id)
                    .with_for_update()
                )
                if claimed_image and claimed_image.status == "segmented":
                    claimed_image.status = "pending"
                    db.commit()
        raise
    finally:
        engine.dispose()
