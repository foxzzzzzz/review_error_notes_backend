from copy import deepcopy
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, or_, select
from sqlalchemy.orm import aliased
from app.database import get_db
from app.api.deps import get_default_student
from app.models.student import Student
from app.models.wrong_question import WrongQuestion
from app.models.wrong_image import WrongImage
from app.schemas.question import (
    QuestionOut,
    QuestionUpdate,
    ReviewDecisionRequest,
    ReviewImageReprocessRequest,
    ManualWrongQuestionRequest,
    ManualQuestionSuggestionRequest,
)
from app.config import settings
from app.tasks.process_image import process_image
from app.services.chinese_marked_evidence import PIPELINE_NAME
from app.services.question_image import (
    QuestionImageInvalid,
    QuestionImageNotFound,
    _validated_normalized_bbox,
    render_question_image,
)
from app.services.manual_question_suggestion import recognize_manual_suggestion

router = APIRouter(prefix="/questions", tags=["questions"])


def _normalize_created_from(created_from: datetime) -> datetime:
    if created_from.tzinfo is None:
        return created_from
    return created_from.astimezone(timezone.utc).replace(tzinfo=None)


def _current_evidence_prompt_values(question, raw_json: dict, bundle: dict) -> dict:
    """Read the current review display without copying observations to legacy keys."""
    previous_override = bundle.get("human_override") or {}
    deepseek_content = raw_json.get("deepseek_content") or {}
    marked_page_item = raw_json.get("marked_page_item") or {}
    fields = bundle.get("fields") or {}

    def selected(field_name: str) -> str:
        field = fields.get(field_name) or {}
        return str(field.get("selected_value") or "").strip()

    question_type = str(question.question_type or "").strip()
    instruction = (
        str(previous_override.get("instruction") or "").strip()
        or selected("printed_instruction")
        or str(deepseek_content.get("instruction") or "").strip()
    )
    prompt_roles = {
        "write_word": ("printed_pinyin", "printed_prompt", "printed_hanzi"),
        "write_pinyin": ("printed_hanzi", "printed_prompt", "printed_pinyin"),
    }.get(
        question_type,
        ("printed_prompt", "printed_pinyin", "printed_hanzi"),
    )
    prompt_text = str(previous_override.get("prompt_text") or "").strip()
    if not prompt_text:
        prompt_text = next(
            (value for value in (selected(role) for role in prompt_roles) if value),
            str(
                deepseek_content.get("prompt_text")
                or marked_page_item.get("printed_question")
                or ""
            ).strip(),
        )
    return {
        "instruction": instruction,
        "prompt_text": prompt_text,
        "question_type": str(
            previous_override.get("question_type") or question_type
        ).strip(),
    }


@router.post("/review/images/{image_id}/manual-suggestion")
async def suggest_manual_review_question(
    image_id: str,
    data: ManualQuestionSuggestionRequest,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    image = await db.scalar(
        select(WrongImage).where(
            WrongImage.id == image_id,
            WrongImage.student_id == student.id,
        )
    )
    if image is None:
        raise HTTPException(status_code=404, detail="Review image not found")
    if image.status not in {"needs_review", "confirmed"}:
        raise HTTPException(status_code=409, detail="Image is not available for review")

    try:
        bbox = _validated_normalized_bbox(data.bbox)
    except QuestionImageInvalid as exc:
        raise HTTPException(
            status_code=422,
            detail="bbox must be normalized left, top, right, bottom",
        ) from exc

    image_path = Path(settings.UPLOAD_DIR) / Path(image.original_url).name
    try:
        return await asyncio.to_thread(
            recognize_manual_suggestion,
            str(image_path),
            bbox,
            data.mode,
            subject=image.subject,
            grade=image.grade,
            semester=image.semester,
        )
    except Exception as exc:
        # Suggestions are optional; the client can continue with manual entry.
        raise HTTPException(
            status_code=503,
            detail="Recognition suggestion unavailable; continue manual entry",
        ) from exc


@router.post("/review/images/{image_id}/manual-questions")
async def add_manual_review_question(
    image_id: str,
    data: ManualWrongQuestionRequest,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    image = await db.scalar(
        select(WrongImage)
        .where(WrongImage.id == image_id, WrongImage.student_id == student.id)
        .with_for_update()
    )
    if image is None:
        raise HTTPException(status_code=404, detail="Review image not found")
    if image.status not in {"needs_review", "confirmed"}:
        raise HTTPException(status_code=409, detail="Image is not available for review")

    try:
        bbox = _validated_normalized_bbox(data.bbox)
    except QuestionImageInvalid as exc:
        raise HTTPException(status_code=422, detail="bbox must be normalized left, top, right, bottom") from exc

    question_type = data.question_type.strip()
    if question_type not in {"write_pinyin", "write_word", "fill_blank", "calculation", "other"}:
        raise HTTPException(status_code=422, detail="Unsupported question_type")
    values = {
        "bbox": bbox,
        "instruction": data.instruction.strip(),
        "prompt_text": data.prompt_text.strip(),
        "question_type": question_type,
        "correct_answer": data.correct_answer.strip(),
        "student_answer": (data.student_answer or "").strip() or None,
    }
    if not all(values[key] for key in ("instruction", "prompt_text", "correct_answer")):
        raise HTTPException(status_code=422, detail="Question fields must not be blank")

    existing = await db.get(WrongQuestion, data.question_id)
    if existing is not None:
        previous = ((existing.ocr_raw_json or {}).get("evidence_bundle") or {}).get(
            "manual_add_request"
        )
        if (
            existing.image_id == image.id
            and existing.student_id == student.id
            and previous == values
        ):
            return {"question_id": str(existing.id), "idempotent": True}
        raise HTTPException(status_code=409, detail="question_id was already used")

    raw_json = {
        "evidence_bundle": {
            "human_confirmed_prompt": {
                "instruction": values["instruction"],
                "prompt_text": values["prompt_text"],
                "question_type": values["question_type"],
                "source": "human",
                "actor_id": str(student.id),
            },
            "review_history": [
                {
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    "source": "human",
                    "actor_id": str(student.id),
                    "decision": "manual_add",
                    "new_values": values,
                }
            ],
            "manual_add_request": values,
        }
    }
    question = WrongQuestion(
        id=data.question_id,
        student_id=student.id,
        image_id=image.id,
        crop_region={"bbox_format": "normalized_ltrb", "bbox": bbox},
        subject=image.subject,
        grade=image.grade,
        semester=image.semester,
        ocr_text=values["student_answer"],
        ocr_answer=values["correct_answer"],
        ocr_raw_json=raw_json,
        recognition_pipeline=PIPELINE_NAME,
        mark_status="confirmed",
        question_evidence_status="confirmed",
        answer_status="confirmed",
        question_type=question_type,
        tags=[],
        collection_status="collected",
        review_status="confirmed",
        mastery_status="learning",
    )
    db.add(question)
    image.question_count = (image.question_count or 0) + 1
    pending = await db.scalar(
        select(WrongQuestion.id)
        .where(
            WrongQuestion.image_id == image.id,
            WrongQuestion.student_id == student.id,
            WrongQuestion.deleted_at.is_(None),
            WrongQuestion.collection_status == "pending_review",
        )
        .limit(1)
    )
    if pending is None:
        image.status = "confirmed"
        image.error_code = None
        image.error_message = None
    await db.commit()
    return {"question_id": str(question.id), "idempotent": False}


@router.get("", response_model=list[QuestionOut])
async def list_questions(
    subject: str = None,
    grade: int = None,
    semester: int = None,
    status: str = None,
    mastery_status: Literal["learning", "mastered"] | None = Query(None),
    tag: str = None,
    keyword: Annotated[str | None, Query(max_length=100)] = None,
    limit: int = Query(20, le=100),
    offset: int = 0,
    created_from: datetime = None,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    q = select(WrongQuestion).where(
        WrongQuestion.student_id == student.id,
        WrongQuestion.deleted_at.is_(None),
        WrongQuestion.collection_status == "collected",
    )
    if subject:
        q = q.where(WrongQuestion.subject == subject)
    if grade:
        q = q.where(WrongQuestion.grade == grade)
    if semester:
        q = q.where(WrongQuestion.semester == semester)
    if status:
        q = q.where(WrongQuestion.review_status == status)
    if mastery_status:
        q = q.where(WrongQuestion.mastery_status == mastery_status)
    if tag:
        q = q.where(WrongQuestion.tags.any(tag))
    if keyword and keyword.strip():
        escaped_keyword = (
            keyword.strip()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        pattern = f"%{escaped_keyword}%"
        q = q.where(
            or_(
                WrongQuestion.ocr_text.ilike(pattern, escape="\\"),
                WrongQuestion.ocr_answer.ilike(pattern, escape="\\"),
                func.array_to_string(WrongQuestion.tags, " ").ilike(
                    pattern,
                    escape="\\",
                ),
            )
        )
    if created_from:
        q = q.where(WrongQuestion.created_at >= _normalize_created_from(created_from))
    q = q.order_by(
        WrongQuestion.created_at.desc(),
        WrongQuestion.id.desc(),
    ).offset(offset).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/review/images")
async def list_review_images(
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    collected_question = aliased(WrongQuestion)
    auto_collected_count = (
        select(func.count(collected_question.id))
        .where(
            collected_question.image_id == WrongImage.id,
            collected_question.deleted_at.is_(None),
            collected_question.collection_status == "collected",
        )
        .correlate(WrongImage)
        .scalar_subquery()
    )
    result = await db.execute(
        select(WrongQuestion, WrongImage, auto_collected_count.label("auto_collected_count"))
        .join(WrongImage, WrongImage.id == WrongQuestion.image_id)
        .where(
            WrongQuestion.student_id == student.id,
            WrongQuestion.deleted_at.is_(None),
            WrongQuestion.collection_status == "pending_review",
        )
        .order_by(WrongImage.created_at.desc(), WrongQuestion.created_at.asc())
    )
    groups = {}
    for question, image, collected_count in result.all():
        image_id = str(image.id)
        group = groups.setdefault(
            image_id,
            {
                "group_type": "questions",
                "image_id": image_id,
                "question_count": 0,
                "auto_collected_count": collected_count,
                "issue_code": None,
                "issue_message": None,
                "questions": [],
                "_created_at": image.created_at,
            },
        )
        item = QuestionOut.model_validate(question).model_dump(mode="json")
        item["crop_region"] = question.crop_region
        item["review_fields"] = _current_evidence_prompt_values(
            question,
            question.ocr_raw_json or {},
            (question.ocr_raw_json or {}).get("evidence_bundle") or {},
        )
        group["questions"].append(item)
        group["question_count"] += 1

    issue_result = await db.execute(
        select(WrongImage, auto_collected_count.label("auto_collected_count"))
        .where(
            WrongImage.student_id == student.id,
            WrongImage.status == "needs_review",
            WrongImage.question_count == 0,
            WrongImage.error_code.is_not(None),
        )
        .order_by(WrongImage.created_at.desc())
        .limit(settings.INCOMPLETE_IMAGE_STATUS_LIMIT)
    )
    for image, collected_count in issue_result.all():
        image_id = str(image.id)
        groups.setdefault(
            image_id,
            {
                "group_type": "image_issue",
                "image_id": image_id,
                "question_count": 0,
                "auto_collected_count": collected_count,
                "issue_code": image.error_code,
                "issue_message": image.error_message,
                "questions": [],
                "_created_at": image.created_at,
            },
        )
    completed_result = await db.execute(
        select(WrongImage, auto_collected_count.label("auto_collected_count"))
        .where(
            WrongImage.student_id == student.id,
            WrongImage.status == "confirmed",
        )
        .order_by(WrongImage.created_at.desc())
        .limit(settings.REVIEW_IMAGE_HISTORY_LIMIT)
    )
    for image, collected_count in completed_result.all():
        image_id = str(image.id)
        groups.setdefault(
            image_id,
            {
                "group_type": "completed_image",
                "image_id": image_id,
                "question_count": 0,
                "auto_collected_count": collected_count,
                "issue_code": None,
                "issue_message": None,
                "questions": [],
                "_created_at": image.created_at,
            },
        )
    ordered_groups = sorted(
        groups.values(), key=lambda group: group["_created_at"], reverse=True
    )
    for group in ordered_groups:
        group.pop("_created_at", None)
    return ordered_groups


@router.post("/review/images/{image_id}/decisions")
async def decide_image_reviews(
    image_id: str,
    data: ReviewDecisionRequest,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    decisions = {str(item.question_id): item for item in data.decisions}
    if len(decisions) != len(data.decisions):
        raise HTTPException(status_code=400, detail="Duplicate question decisions")
    image = await db.scalar(
        select(WrongImage)
        .where(
            WrongImage.id == image_id,
            WrongImage.student_id == student.id,
        )
        .with_for_update()
    )
    if not image:
        raise HTTPException(status_code=404, detail="Review image not found")
    result = await db.execute(
        select(WrongQuestion)
        .where(
            WrongQuestion.id.in_(list(decisions)),
            WrongQuestion.image_id == image_id,
            WrongQuestion.student_id == student.id,
            WrongQuestion.deleted_at.is_(None),
        )
        .with_for_update()
    )
    questions = result.scalars().all()
    if len(questions) != len(decisions):
        raise HTTPException(status_code=404, detail="Review question not found")
    pending_questions = []
    for question in questions:
        decision = decisions[str(question.id)]
        if (
            question.recognition_pipeline == PIPELINE_NAME
            and decision.decision == "collect"
            and any(
                value is not None and not value.strip()
                for value in (
                    decision.question_type,
                    decision.instruction,
                    decision.prompt_text,
                )
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="prompt corrections must be non-blank when provided",
            )
        if question.collection_status == "pending_review":
            if (
                question.recognition_pipeline == PIPELINE_NAME
                and decision.decision == "collect"
                and not (decision.correct_answer or "").strip()
            ):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="correct_answer is required when collecting a question",
                )
            if (
                question.recognition_pipeline == PIPELINE_NAME
                and decision.decision == "collect"
            ):
                raw_json = question.ocr_raw_json or {}
                bundle = raw_json.get("evidence_bundle") or {}
                current_prompt = _current_evidence_prompt_values(
                    question,
                    raw_json,
                    bundle,
                )
                final_prompt = {
                    "instruction": (
                        decision.instruction.strip()
                        if decision.instruction is not None
                        else current_prompt["instruction"]
                    ),
                    "prompt_text": (
                        decision.prompt_text.strip()
                        if decision.prompt_text is not None
                        else current_prompt["prompt_text"]
                    ),
                    "question_type": (
                        decision.question_type.strip()
                        if decision.question_type is not None
                        else current_prompt["question_type"]
                    ),
                }
                if not all(final_prompt.values()):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="a complete current prompt is required when collecting a question",
                    )
            pending_questions.append(question)
        else:
            if question.collection_status == "superseded":
                raise HTTPException(status_code=409, detail="Review question was superseded")
            if question.recognition_pipeline != PIPELINE_NAME:
                raise HTTPException(status_code=404, detail="Review question not found")
            bundle = (question.ocr_raw_json or {}).get("evidence_bundle", {})
            history = bundle.get("review_history", [])
            last_review = history[-1] if history else None
            correct_answer = (
                decision.correct_answer.strip()
                if decision.correct_answer is not None
                else None
            )
            corrections = {
                key: value.strip()
                for key, value in (
                    ("question_type", decision.question_type),
                    ("instruction", decision.instruction),
                    ("prompt_text", decision.prompt_text),
                )
                if value is not None
            }
            new_values = last_review.get("new_values", {}) if last_review else {}
            if not (
                question.collection_status
                == ("collected" if decision.decision == "collect" else "ignored")
                and question.review_status == "confirmed"
                and last_review
                and last_review.get("decision") == decision.decision
                and new_values.get("correct_answer") == correct_answer
                and all(new_values.get(key) == corrections.get(key) for key in (
                    "question_type",
                    "instruction",
                    "prompt_text",
                ))
            ):
                raise HTTPException(status_code=409, detail="Review decision conflicts with terminal state")
    if not pending_questions:
        remaining = await db.scalar(
            select(WrongQuestion.id)
            .where(
                WrongQuestion.image_id == image.id,
                WrongQuestion.collection_status == "pending_review",
                WrongQuestion.deleted_at.is_(None),
            )
            .limit(1)
        )
        return {
            "collected": sum(item.decision == "collect" for item in decisions.values()),
            "ignored": sum(item.decision == "ignore" for item in decisions.values()),
            "remaining": bool(remaining),
        }
    for question in pending_questions:
        decision = decisions[str(question.id)]
        if question.recognition_pipeline == PIPELINE_NAME:
            correct_answer = (
                decision.correct_answer.strip()
                if decision.correct_answer is not None
                else None
            )
            corrections = {
                key: value.strip()
                for key, value in (
                    ("question_type", decision.question_type),
                    ("instruction", decision.instruction),
                    ("prompt_text", decision.prompt_text),
                )
                if value is not None
            }
            raw_json = deepcopy(question.ocr_raw_json or {})
            bundle = deepcopy(raw_json.get("evidence_bundle") or {})
            previous_override = deepcopy(bundle.get("human_override") or {})
            current_prompt = _current_evidence_prompt_values(
                question,
                raw_json,
                bundle,
            )
            old_values = {
                "correct_answer": question.ocr_answer,
                "question_type": question.question_type,
                "instruction": current_prompt["instruction"] or None,
                "prompt_text": current_prompt["prompt_text"] or None,
            }
            if corrections:
                bundle["human_override"] = {**previous_override, **corrections}
            if decision.decision == "collect":
                confirmed_prompt = {
                    "instruction": corrections.get("instruction")
                    or current_prompt["instruction"],
                    "prompt_text": corrections.get("prompt_text")
                    or current_prompt["prompt_text"],
                    "question_type": corrections.get("question_type")
                    or current_prompt["question_type"],
                    "source": "human",
                    "actor_id": str(student.id),
                }
                if not all(
                    confirmed_prompt[key]
                    for key in ("instruction", "prompt_text", "question_type")
                ):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="a complete current prompt is required when collecting a question",
                    )
                bundle["human_confirmed_prompt"] = confirmed_prompt
                question.ocr_answer = correct_answer
                question.question_type = confirmed_prompt["question_type"]
                question.mark_status = "confirmed"
                question.question_evidence_status = "confirmed"
                question.answer_status = "confirmed"
                question.collection_status = "collected"
            else:
                question.collection_status = "ignored"
            question.review_status = "confirmed"
            bundle.setdefault("review_history", []).append(
                {
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    "source": "human",
                    "actor_id": str(student.id),
                    "decision": decision.decision,
                    "old_values": old_values,
                    "new_values": {
                        "correct_answer": correct_answer,
                        "question_type": corrections.get("question_type"),
                        "instruction": corrections.get("instruction"),
                        "prompt_text": corrections.get("prompt_text"),
                    },
                }
            )
            raw_json["evidence_bundle"] = bundle
            question.ocr_raw_json = raw_json
        else:
            question.collection_status = (
                "collected" if decision.decision == "collect" else "ignored"
            )
            question.review_status = "confirmed"

    await db.flush()
    remaining = await db.scalar(
        select(WrongQuestion.id)
        .where(
            WrongQuestion.image_id == image.id,
            WrongQuestion.collection_status == "pending_review",
            WrongQuestion.deleted_at.is_(None),
        )
        .limit(1)
    )
    if not remaining:
        image.status = "confirmed"
    await db.commit()
    return {
        "collected": sum(item.decision == "collect" for item in decisions.values()),
        "ignored": sum(item.decision == "ignore" for item in decisions.values()),
        "remaining": bool(remaining),
    }


@router.post("/review/images/{image_id}/reprocess")
async def reprocess_review_image(
    image_id: str,
    data: ReviewImageReprocessRequest,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    image = await db.scalar(
        select(WrongImage)
        .where(
            WrongImage.id == image_id,
            WrongImage.student_id == student.id,
        )
        .with_for_update()
    )
    if not image:
        raise HTTPException(status_code=404, detail="Review image not found")
    result = await db.execute(
        select(WrongQuestion)
        .where(
            WrongQuestion.image_id == image.id,
            WrongQuestion.student_id == student.id,
            WrongQuestion.collection_status == "pending_review",
            WrongQuestion.deleted_at.is_(None),
        )
        .with_for_update()
    )
    questions = result.scalars().all()
    actionable_image_issue = bool(
        image.status == "needs_review"
        and image.question_count == 0
        and image.error_code
    )
    if not questions and not actionable_image_issue:
        raise HTTPException(status_code=409, detail="No pending review questions")
    previous_status = image.status
    previous_question_count = image.question_count
    previous_error_code = image.error_code
    previous_error_message = image.error_message
    previous_correction = image.recognition_correction
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for question in questions:
        question.collection_status = "superseded"
        question.deleted_at = now
    image.status = "pending"
    image.question_count = 0
    image.error_code = None
    image.error_message = None
    image.recognition_correction = data.correction
    await db.commit()
    try:
        filepath = str(Path(settings.UPLOAD_DIR) / Path(image.original_url).name)
        process_image.delay(str(image.id), filepath)
    except Exception:
        for question in questions:
            question.collection_status = "pending_review"
            question.deleted_at = None
        image.status = previous_status
        image.question_count = previous_question_count
        image.error_code = previous_error_code
        image.error_message = previous_error_message
        image.recognition_correction = previous_correction
        await db.commit()
        raise HTTPException(status_code=503, detail="处理任务投递失败，请重试")
    return {"image_id": str(image.id), "status": "pending"}


@router.get("/{question_id}", response_model=QuestionOut)
async def get_question(
    question_id: str,
    student: Student = Depends(get_default_student),
    db=Depends(get_db),
):
    result = await db.execute(
        select(WrongQuestion, WrongImage)
        .join(WrongImage, WrongImage.id == WrongQuestion.image_id)
        .where(
            WrongQuestion.id == question_id,
            WrongQuestion.student_id == student.id,
            WrongQuestion.deleted_at.is_(None),
        )
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    q, image = row
    data = QuestionOut.model_validate(q).model_dump()
    data["crop_region"] = q.crop_region
    return data


@router.get("/{question_id}/image")
async def get_question_image(
    question_id: str,
    view: Literal["crop", "original"] = "crop",
    student: Student = Depends(get_default_student),
    db=Depends(get_db),
):
    result = await db.execute(
        select(WrongQuestion, WrongImage)
        .join(WrongImage, WrongImage.id == WrongQuestion.image_id)
        .where(
            WrongQuestion.id == question_id,
            WrongQuestion.student_id == student.id,
            WrongQuestion.deleted_at.is_(None),
        )
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Question image not found")

    question, image = row
    image_path = Path(settings.UPLOAD_DIR) / Path(image.original_url).name
    try:
        content = render_question_image(
            image_path,
            question.crop_region,
            view,
            settings.MINIMAX_IMAGE_JPEG_QUALITY,
            settings.QUESTION_IMAGE_MAX_PIXELS,
        )
    except QuestionImageNotFound:
        raise HTTPException(status_code=404, detail="Question image not found")
    except QuestionImageInvalid:
        raise HTTPException(status_code=422, detail="Question image is invalid")

    return Response(content=content, media_type="image/jpeg")


@router.patch("/{question_id}")
async def update_question(
    question_id: str,
    data: QuestionUpdate,
    student: Student = Depends(get_default_student),
    db=Depends(get_db),
):
    result = await db.execute(
        select(WrongQuestion).where(
            WrongQuestion.id == question_id,
            WrongQuestion.student_id == student.id,
            WrongQuestion.deleted_at.is_(None),
        )
    )
    q = result.scalar_one_or_none()
    if not q:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    if (
        q.recognition_pipeline == PIPELINE_NAME
        and data.model_dump(exclude_unset=True).get("review_status") == "confirmed"
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Use the review-decision endpoint to confirm this question",
        )
    if (
        q.recognition_pipeline == PIPELINE_NAME
        and "ocr_text" in data.model_dump(exclude_unset=True)
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Student handwriting must not be overwritten through this endpoint",
        )
    if q.recognition_pipeline == PIPELINE_NAME:
        for k, v in data.model_dump(exclude_unset=True).items():
            setattr(q, k, v)
        await db.commit()
        return {"ok": True}
    was_needs_review = q.review_status == "needs_review"
    image = None
    if was_needs_review:
        image = await db.scalar(
            select(WrongImage)
            .where(WrongImage.id == q.image_id)
            .with_for_update()
        )
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(q, k, v)
    if was_needs_review:
        q.review_status = "confirmed"
        await db.flush()
        remaining_review = await db.scalar(
            select(WrongQuestion.id)
            .where(
                WrongQuestion.image_id == q.image_id,
                WrongQuestion.review_status == "needs_review",
                WrongQuestion.deleted_at.is_(None),
            )
            .limit(1)
        )
        if not remaining_review and image:
            image.status = "confirmed"
    await db.commit()
    return {"ok": True}


@router.delete("/{question_id}")
async def delete_question(
    question_id: str,
    student: Student = Depends(get_default_student),
    db=Depends(get_db),
):
    result = await db.execute(
        select(WrongQuestion).where(
            WrongQuestion.id == question_id,
            WrongQuestion.student_id == student.id,
        )
    )
    q = result.scalar_one_or_none()
    if not q:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    if q.deleted_at is None:
        q.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()
    return {"ok": True}
