"""Celery task for MiniMax multimodal wrong-question recognition."""

import json
import logging

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import settings
# Register the complete SQLAlchemy foreign-key graph before sync-session flushes.
from app.models.practice_sheet import PracticeSheet  # noqa: F401
from app.models.sheet_item import SheetItem  # noqa: F401
from app.models.student import Student  # noqa: F401
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion
from app.services.local_ocr_verification import RapidOCRVerifier
from app.services.vision_recognition import (
    MiniMaxVisionClient,
    image_status_for,
    recognize_question_batch,
)
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def log_mark_validation_diagnostics(
    image_id: str,
    question_values: list[dict],
) -> None:
    diagnostics = (
        question_values[0]["ocr_raw_json"].get("error_mark_validation", [])
        if question_values
        else []
    )
    logger.info(
        "error_mark_validation image_id=%s diagnostics=%s",
        image_id,
        json.dumps(diagnostics, ensure_ascii=False, separators=(",", ":")),
    )


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

        result, question_values = recognize_question_batch(
            client=MiniMaxVisionClient.from_settings(),
            image_path=filepath,
            subject_hint=subject_hint,
            confidence_threshold=settings.MINIMAX_CONFIDENCE_THRESHOLD,
            mark_confidence_threshold=settings.MINIMAX_MARK_CONFIDENCE_THRESHOLD,
            localization_threshold=settings.MINIMAX_LOCALIZATION_CONFIDENCE_THRESHOLD,
            localization_max_area_ratio=settings.MINIMAX_LOCALIZATION_MAX_AREA_RATIO,
            crop_context_padding_ratio=settings.QUESTION_CROP_CONTEXT_PADDING_RATIO,
            red_pixel_min_ratio=settings.MARK_RED_PIXEL_MIN_RATIO,
            red_pixel_expansion_ratio=settings.MARK_RED_PIXEL_EXPANSION_RATIO,
            tag_config_path=settings.TAG_ALIAS_CONFIG_PATH,
            ocr_verifier=RapidOCRVerifier(
                enabled=settings.LOCAL_OCR_ENABLED,
                library_version=settings.LOCAL_OCR_VERSION,
                engine_name=settings.LOCAL_OCR_ENGINE,
                model_version=settings.LOCAL_OCR_MODEL_VERSION,
                model_type=settings.LOCAL_OCR_MODEL_TYPE,
                model_path=settings.LOCAL_OCR_MODEL_PATH,
                max_pixels=settings.QUESTION_IMAGE_MAX_PIXELS,
                line_confidence_threshold=settings.LOCAL_OCR_LINE_CONFIDENCE_THRESHOLD,
                min_effective_characters=settings.LOCAL_OCR_MIN_EFFECTIVE_CHARACTERS,
                support_similarity_threshold=settings.LOCAL_OCR_SUPPORT_SIMILARITY_THRESHOLD,
                contradiction_similarity_threshold=settings.LOCAL_OCR_CONTRADICTION_SIMILARITY_THRESHOLD,
            ),
        )
        log_mark_validation_diagnostics(image_id, question_values)

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
