"""Celery task for MiniMax multimodal wrong-question recognition."""

import json
import logging
import unicodedata

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import settings
# Register the complete SQLAlchemy foreign-key graph before sync-session flushes.
from app.models.account import Account
from app.models.practice_sheet import PracticeSheet  # noqa: F401
from app.models.sheet_item import SheetItem  # noqa: F401
from app.models.student import Student  # noqa: F401
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion
from app.services.local_ocr_verification import RapidOCRVerifier
from app.services.vision_recognition import (
    MiniMaxVisionClient,
    VisionRecognitionError,
    image_status_for,
    recognize_question_batch,
    safe_recognition_diagnostic,
)
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

def processing_failure_for(error: Exception) -> tuple[str, str]:
    if isinstance(error, VisionRecognitionError):
        return error.code, error.user_message
    return "recognition_internal_error", "识别暂时失败，请稍后重试"


def collection_status_to_persist(collection_status: str) -> str:
    """Keep recognized candidates available for an explicit user decision."""
    return "pending_review" if collection_status == "ignored" else collection_status


def should_persist_candidate(values: dict, recognition_correction: str | None) -> bool:
    """Strict correction modes discard candidates previously judged as correct."""
    return not (
        recognition_correction in {"false_positives", "both"}
        and values["collection_status"] == "ignored"
    )


def candidate_identity_for(values: dict) -> tuple[str, str, str, str, str]:
    def normalize(value) -> str:
        normalized = unicodedata.normalize("NFKC", value or "").casefold()
        return "".join(character for character in normalized if character.isalnum())

    raw = values.get("ocr_raw_json", {})
    return (
        normalize(raw.get("instruction")),
        normalize(raw.get("prompt_text")),
        normalize(values.get("ocr_text")),
        normalize(values.get("ocr_answer")),
        values.get("question_type") or "",
    )


def discard_pending_duplicates_of_collected(question_values: list[dict]) -> list[dict]:
    """Prefer a reliable auto-collected copy over the same pending candidate."""
    collected_identities = {
        candidate_identity_for(candidate)
        for candidate in question_values
        if candidate["collection_status"] == "collected"
    }
    return [
        candidate
        for candidate in question_values
        if not (
            candidate["collection_status"] in {"pending_review", "ignored"}
            and candidate_identity_for(candidate) in collected_identities
        )
    ]


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

    claimed = False
    stage = "claim"
    try:
        with Session(engine) as db:
            image = db.scalar(
                select(WrongImage)
                .join(Student, Student.id == WrongImage.student_id)
                .join(Account, Account.id == Student.account_id)
                .where(WrongImage.id == image_id)
                .where(Account.status == "active")
                .with_for_update()
            )
            if not image or image.status != "pending":
                return
            subject_hint = image.subject
            recognition_correction = image.recognition_correction
            image.status = "segmented"
            db.commit()
            claimed = True

        stage = "recognition"
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
            recognition_correction=recognition_correction,
        )
        log_mark_validation_diagnostics(image_id, question_values)
        question_values = discard_pending_duplicates_of_collected(question_values)

        stage = "persistence"
        with Session(engine) as db:
            image = db.scalar(
                select(WrongImage)
                .join(Student, Student.id == WrongImage.student_id)
                .join(Account, Account.id == Student.account_id)
                .where(WrongImage.id == image_id)
                .where(Account.status == "active")
                .with_for_update()
            )
            if not image or image.status != "segmented":
                return

            persisted_values = []
            for values in question_values:
                values["ocr_raw_json"]["ignored_text"] = result.ignored_text
                if not should_persist_candidate(values, recognition_correction):
                    continue
                collection_status = collection_status_to_persist(
                    values["collection_status"]
                )
                values["collection_status"] = collection_status
                question_values_for_db = {
                    key: value
                    for key, value in values.items()
                    if key != "collection_status"
                }
                question = WrongQuestion(
                    student_id=image.student_id,
                    image_id=image.id,
                    grade=image.grade,
                    semester=image.semester,
                    collection_status=collection_status,
                    **question_values_for_db,
                )
                db.add(question)
                persisted_values.append(question_values_for_db)

            image.question_count = len(persisted_values)
            image.status = image_status_for(question_values)
            image.recognition_correction = None
            if not image.subject:
                image.subject = result.items[0].subject
            db.commit()
            claimed = False
    except Exception as exc:
        if claimed:
            with Session(engine) as db:
                claimed_image = db.scalar(
                    select(WrongImage)
                    .where(WrongImage.id == image_id)
                    .with_for_update()
                )
                if claimed_image and claimed_image.status == "segmented":
                    claimed_image.status = "failed"
                    error_code, error_message = processing_failure_for(exc)
                    claimed_image.error_code = error_code
                    claimed_image.error_message = error_message
                    db.commit()
        logger.exception(
            "image_recognition_failed image_id=%s stage=%s error_code=%s error_type=%s diagnostic=%s",
            image_id,
            stage,
            processing_failure_for(exc)[0],
            type(exc).__name__,
            json.dumps(
                safe_recognition_diagnostic(exc),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    finally:
        engine.dispose()
