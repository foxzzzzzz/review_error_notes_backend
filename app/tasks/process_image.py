"""Celery task for configured multimodal wrong-question recognition."""

import json
import logging
import math
import time
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
from app.services.error_mark_validation import scan_red_mark_regions
from app.services.local_ocr_verification import RapidOCRVerifier
from app.services.chinese_marked_evidence import PIPELINE_NAME
from app.services.vision_provider import create_vision_client
from app.services.vision_recognition import (
    ImageReviewRequired,
    VisionRecognitionError,
    image_status_for,
    recognize_question_batch,
    safe_recognition_diagnostic,
)
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def evidence_mode_requested_for(
    subject_hint: str | None,
    recognition_correction: str | None,
) -> bool:
    if (
        not settings.CHINESE_MARKED_EVIDENCE_ENABLED
        or recognition_correction == "force_unmarked"
        or not isinstance(subject_hint, str)
    ):
        return False
    configured_subjects = {
        subject.strip().casefold()
        for subject in settings.CHINESE_MARKED_EVIDENCE_SUBJECTS.split(",")
        if subject.strip()
    }
    return subject_hint.strip().casefold() in configured_subjects


def evidence_timing_for(
    recognition_started_at: float,
    observed_at: float,
    timeout_seconds: float,
) -> dict:
    elapsed_seconds = max(0.0, observed_at - recognition_started_at)
    return {
        "timeout_seconds": float(timeout_seconds),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "remaining_seconds": round(max(0.0, timeout_seconds - elapsed_seconds), 6),
        "exhausted": observed_at >= recognition_started_at + timeout_seconds,
    }

def processing_failure_for(error: Exception) -> tuple[str, str]:
    if isinstance(error, VisionRecognitionError):
        return error.code, error.user_message
    return "recognition_internal_error", "识别暂时失败，请稍后重试"


def should_persist_candidate(values: dict, recognition_correction: str | None) -> bool:
    """Never persist candidates explicitly discarded by the evidence policy."""
    return values["collection_status"] != "ignored"


def candidate_identity_for(values: dict) -> tuple:
    def normalize(value) -> str:
        normalized = unicodedata.normalize("NFKC", value or "").casefold()
        return "".join(character for character in normalized if character.isalnum())

    if values.get("recognition_pipeline") == PIPELINE_NAME:
        raw = values.get("ocr_raw_json")
        bundle = raw.get("evidence_bundle") if isinstance(raw, dict) else None
        identity = bundle.get("identity") if isinstance(bundle, dict) else None
        if not isinstance(identity, dict):
            raise ValueError("evidence identity must be an object")
        image_id = identity.get("image_id")
        mark_id = identity.get("mark_id")
        geometry = identity.get("question_geometry")
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, (str, int))
            or (isinstance(image_id, str) and not image_id.strip())
            or isinstance(mark_id, bool)
            or not isinstance(mark_id, int)
            or mark_id < 0
            or not isinstance(geometry, dict)
            or not geometry
        ):
            raise ValueError("evidence identity has invalid image, mark, or geometry")

        def validate_bbox(value) -> None:
            if value is None:
                return
            if not isinstance(value, list) or len(value) != 4:
                raise ValueError("evidence identity bbox must contain four coordinates")
            if any(
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(coordinate)
                for coordinate in value
            ):
                raise ValueError("evidence identity bbox coordinates must be finite")
            left, top, right, bottom = (float(coordinate) for coordinate in value)
            if (
                not all(0.0 <= coordinate <= 1.0 for coordinate in (left, top, right, bottom))
                or right <= left
                or bottom <= top
            ):
                raise ValueError("evidence identity bbox must be normalized with positive area")

        for key in ("bbox", "question_bbox", "answer_bbox", "prompt_bbox"):
            if key in geometry:
                validate_bbox(geometry[key])
        mark_geometry = geometry.get("mark")
        if mark_geometry is not None:
            if not isinstance(mark_geometry, dict):
                raise ValueError("evidence identity mark geometry must be an object")
            for key in ("bbox", "cross_bbox", "circle_bbox"):
                if key in mark_geometry:
                    validate_bbox(mark_geometry[key])

        def canonicalize(value):
            if value is None or isinstance(value, (str, bool)):
                return value
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError("evidence identity geometry must be finite")
                if value == 0 or value.is_integer():
                    return int(value)
                return value
            if isinstance(value, list):
                return [canonicalize(item) for item in value]
            if isinstance(value, dict) and all(
                isinstance(key, str) for key in value
            ):
                return {
                    key: canonicalize(item)
                    for key, item in sorted(value.items())
                }
            raise ValueError("evidence identity geometry is not JSON-compatible")

        canonical_geometry = json.dumps(
            canonicalize(geometry),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return (PIPELINE_NAME, str(image_id), mark_id, canonical_geometry)

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
    kept = []
    seen_evidence_identities = set()
    for candidate in question_values:
        identity = candidate_identity_for(candidate)
        if candidate.get("recognition_pipeline") == PIPELINE_NAME:
            if identity in seen_evidence_identities:
                continue
            seen_evidence_identities.add(identity)
        if (
            candidate["collection_status"] in {"pending_review", "ignored"}
            and identity in collected_identities
        ):
            continue
        kept.append(candidate)
    return kept


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
    localization = (
        question_values[0]["ocr_raw_json"].get("localization_batch_validation")
        if question_values
        else None
    )
    if localization is not None:
        logger.info(
            "localization_validation image_id=%s diagnostic=%s",
            image_id,
            json.dumps(
                localization,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    correction_group = (
        question_values[0]["ocr_raw_json"].get("correction_group_validation")
        if question_values
        else None
    )
    if correction_group is not None:
        logger.info(
            "correction_group_validation image_id=%s diagnostic=%s",
            image_id,
            json.dumps(
                correction_group,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    three_stage = (
        question_values[0]["ocr_raw_json"].get("three_stage")
        if question_values
        else None
    )
    if three_stage is not None:
        logger.info(
            "three_stage_recognition image_id=%s diagnostic=%s",
            image_id,
            json.dumps(three_stage, ensure_ascii=False, separators=(",", ":")),
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
        evidence_mode = evidence_mode_requested_for(
            subject_hint,
            recognition_correction,
        )
        recognition_started_at = time.monotonic()
        deadline = (
            recognition_started_at
            + settings.CHINESE_MARKED_EVIDENCE_PAGE_TIMEOUT_SECONDS
            if evidence_mode
            and settings.CHINESE_MARKED_EVIDENCE_PAGE_DEADLINE_ENABLED
            else None
        )
        local_red_scan = scan_red_mark_regions(
            filepath,
            max_edge=settings.LOCAL_RED_SCAN_MAX_EDGE,
            min_component_pixels=settings.LOCAL_RED_COMPONENT_MIN_PIXELS,
            max_component_area_ratio=settings.LOCAL_RED_COMPONENT_MAX_AREA_RATIO,
            max_thinness_ratio=settings.LOCAL_RED_COMPONENT_MAX_THINNESS_RATIO,
        )
        vision_client = create_vision_client()
        ocr_verifier = RapidOCRVerifier(
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
        )
        result, question_values = recognize_question_batch(
            client=vision_client,
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
            ocr_verifier=ocr_verifier,
            recognition_correction=recognition_correction,
            local_red_scan=local_red_scan,
            mark_mismatch_retry_count=settings.MINIMAX_MARK_MISMATCH_RETRY_COUNT,
            ocr_full_page_max_edge=settings.LOCAL_OCR_FULL_PAGE_MAX_EDGE,
            ocr_crop_recheck_limit=settings.LOCAL_OCR_CROP_RECHECK_LIMIT,
            force_mode=(
                "unmarked" if recognition_correction == "force_unmarked" else None
            ),
            correction_group_enabled=settings.MARK_CORRECTION_GROUP_ENABLED,
            pair_max_distance_ratio=settings.MARK_PAIR_MAX_DISTANCE_RATIO,
            pair_max_relative_distance_ratio=settings.MARK_PAIR_MAX_RELATIVE_DISTANCE_RATIO,
            pair_min_margin_ratio=settings.MARK_PAIR_MIN_MARGIN_RATIO,
            circle_context_padding_ratio=settings.MARK_CIRCLE_CONTEXT_PADDING_RATIO,
            evidence_context_min_width_ratio=settings.MARK_EVIDENCE_CONTEXT_MIN_WIDTH_RATIO,
            evidence_context_min_height_ratio=settings.MARK_EVIDENCE_CONTEXT_MIN_HEIGHT_RATIO,
            answer_min_circle_overlap_ratio=settings.MARK_ANSWER_MIN_CIRCLE_OVERLAP_RATIO,
            answer_min_answer_overlap_ratio=settings.MARK_ANSWER_MIN_ANSWER_OVERLAP_RATIO,
            answer_hard_min_circle_coverage_ratio=settings.MARK_ANSWER_HARD_MIN_CIRCLE_COVERAGE_RATIO,
            answer_max_center_offset_ratio=settings.MARK_ANSWER_MAX_CENTER_OFFSET_RATIO,
            answer_max_overflow_ratio=settings.MARK_ANSWER_MAX_OVERFLOW_RATIO,
            localization_edge_margin_ratio=settings.MARK_LOCALIZATION_EDGE_MARGIN_RATIO,
            local_red_group_max_gap_ratio=settings.LOCAL_RED_GROUP_MAX_GAP_RATIO,
            local_red_group_max_area_ratio=settings.LOCAL_RED_COMPONENT_MAX_AREA_RATIO,
            dedup_iou_threshold=settings.MARK_DEDUP_IOU_THRESHOLD,
            anchor_max_gap_ratio=settings.MARK_ANCHOR_MAX_GAP_RATIO,
            cross_only_max_gap_ratio=settings.MARK_CROSS_ONLY_MAX_GAP_RATIO,
            semantic_retry_count=settings.MINIMAX_LOCALIZATION_SEMANTIC_RETRY_COUNT,
            marked_ocr_recheck_limit=settings.LOCAL_OCR_MARKED_RECHECK_LIMIT,
            local_red_rescue_min_pixels=settings.LOCAL_RED_RESCUE_MIN_PIXELS,
            three_stage_enabled=settings.MINIMAX_THREE_STAGE_RECOGNITION_ENABLED,
            mark_stage_retry_count=settings.MINIMAX_MARK_STAGE_RETRY_COUNT,
            localization_stage_retry_count=settings.MINIMAX_LOCALIZATION_STAGE_RETRY_COUNT,
            content_stage_retry_count=settings.MINIMAX_CONTENT_STAGE_RETRY_COUNT,
            content_batch_size=settings.MINIMAX_CONTENT_BATCH_SIZE,
            image_max_edge=settings.MINIMAX_IMAGE_MAX_EDGE,
            image_jpeg_quality=settings.MINIMAX_IMAGE_JPEG_QUALITY,
            image_max_pixels=settings.QUESTION_IMAGE_MAX_PIXELS,
            evidence_mode=evidence_mode,
            image_id=image_id if evidence_mode else None,
            deadline=deadline,
            evidence_ocr_crop_recheck_limit=(
                settings.CHINESE_MARKED_EVIDENCE_OCR_CROP_RECHECK_LIMIT
                if evidence_mode
                else None
            ),
            evidence_localization_recheck_limit=(
                settings.CHINESE_MARKED_EVIDENCE_LOCALIZATION_RECHECK_LIMIT
                if evidence_mode
                else None
            ),
            stage_audit_enabled=(
                settings.CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED
                if evidence_mode
                else False
            ),
        )
        log_mark_validation_diagnostics(image_id, question_values)
        question_values = discard_pending_duplicates_of_collected(question_values)

        evidence_values = [
            candidate
            for candidate in question_values
            if candidate.get("recognition_pipeline") == PIPELINE_NAME
        ]
        if evidence_values:
            timeout_seconds = settings.CHINESE_MARKED_EVIDENCE_PAGE_TIMEOUT_SECONDS
            timing_diagnostic = evidence_timing_for(
                recognition_started_at,
                time.monotonic(),
                timeout_seconds,
            )
            for evidence_value in evidence_values:
                evidence_value["ocr_raw_json"]["evidence_timing"] = {
                    "before_persistence": dict(timing_diagnostic)
                }

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

            if evidence_values:
                pre_commit_timing = evidence_timing_for(
                    recognition_started_at,
                    time.monotonic(),
                    settings.CHINESE_MARKED_EVIDENCE_PAGE_TIMEOUT_SECONDS,
                )
                for evidence_value in evidence_values:
                    evidence_value["ocr_raw_json"]["evidence_timing"][
                        "pre_commit"
                    ] = dict(pre_commit_timing)

            persisted_values = []
            for values in question_values:
                values["ocr_raw_json"]["ignored_text"] = result.ignored_text
                if not should_persist_candidate(values, recognition_correction):
                    continue
                collection_status = values["collection_status"]
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
            recognition_mode = (
                question_values[0]["ocr_raw_json"].get("recognition_mode", "unknown")
                if question_values
                else "unknown"
            )
            page_diagnostic = (
                question_values[0]["ocr_raw_json"].get("local_ocr_page")
                if question_values
                else None
            )
            crop_ocr_durations = [
                values["ocr_raw_json"].get("local_ocr", {}).get("duration_ms", 0)
                for values in question_values
            ]
            ocr_calls = sum(duration > 0 for duration in crop_ocr_durations)
            ocr_ms = sum(crop_ocr_durations)
            three_stage_diagnostic = (
                question_values[0]["ocr_raw_json"].get("three_stage", {})
                if question_values
                else {}
            )
            vision_llm_ms = sum(
                float(three_stage_diagnostic.get(key, 0) or 0)
                for key in ("mark_llm_ms", "localization_llm_ms", "content_llm_ms")
            )
            if page_diagnostic and page_diagnostic.get("status") == "available":
                ocr_calls += 1
                ocr_ms += page_diagnostic.get("duration_ms", 0)
            final_status = image.status
            image.error_code = None
            image.error_message = None
            image.recognition_correction = None
            if not image.subject and result.items:
                image.subject = result.items[0].subject
            db.commit()
            recognition_to_commit_timing = (
                evidence_timing_for(
                    recognition_started_at,
                    time.monotonic(),
                    settings.CHINESE_MARKED_EVIDENCE_PAGE_TIMEOUT_SECONDS,
                )
                if evidence_values
                else None
            )
            claimed = False
        if recognition_to_commit_timing is not None:
            logger.info(
                "evidence_recognition_to_commit image_id=%s diagnostic=%s",
                image_id,
                json.dumps(
                    recognition_to_commit_timing,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        logger.info(
            "image_recognition_summary image_id=%s mode=%s pipeline=%s red_scan_ms=%.2f vision_llm_ms=%.2f ocr_ms=%.2f ocr_calls=%s candidate_count=%s persisted_count=%s status=%s",
            image_id,
            recognition_mode,
            (
                question_values[0].get("recognition_pipeline")
                if question_values
                else None
            )
            or three_stage_diagnostic.get("recognition_pipeline", "legacy"),
            local_red_scan.duration_ms,
            vision_llm_ms,
            ocr_ms,
            ocr_calls,
            len(question_values),
            len(persisted_values),
            final_status,
        )
    except ImageReviewRequired as exc:
        if claimed:
            with Session(engine) as db:
                claimed_image = db.scalar(
                    select(WrongImage)
                    .where(WrongImage.id == image_id)
                    .with_for_update()
                )
                if claimed_image and claimed_image.status == "segmented":
                    claimed_image.status = "needs_review"
                    claimed_image.question_count = 0
                    claimed_image.error_code = exc.code
                    claimed_image.error_message = exc.user_message
                    claimed_image.recognition_correction = None
                    db.commit()
        logger.info(
            "image_recognition_needs_review image_id=%s error_code=%s diagnostic=%s",
            image_id,
            exc.code,
            json.dumps(
                safe_recognition_diagnostic(exc),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
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
