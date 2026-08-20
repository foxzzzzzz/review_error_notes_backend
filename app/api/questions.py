from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Query, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
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
)
from app.config import settings
from app.tasks.process_image import process_image
from app.services.question_image import (
    QuestionImageInvalid,
    QuestionImageNotFound,
    render_question_image,
)

router = APIRouter(prefix="/questions", tags=["questions"])


def _normalize_created_from(created_from: datetime) -> datetime:
    if created_from.tzinfo is None:
        return created_from
    return created_from.astimezone(timezone.utc).replace(tzinfo=None)


@router.get("", response_model=list[QuestionOut])
async def list_questions(
    subject: str = None,
    grade: int = None,
    semester: int = None,
    status: str = None,
    mastery_status: Literal["learning", "mastered"] | None = Query(None),
    tag: str = None,
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
    result = await db.execute(
        select(WrongQuestion, WrongImage)
        .join(WrongImage, WrongImage.id == WrongQuestion.image_id)
        .where(
            WrongQuestion.student_id == student.id,
            WrongQuestion.deleted_at.is_(None),
            WrongQuestion.collection_status == "pending_review",
        )
        .order_by(WrongImage.created_at.desc(), WrongQuestion.created_at.asc())
    )
    groups = {}
    for question, image in result.all():
        image_id = str(image.id)
        group = groups.setdefault(
            image_id,
            {
                "image_id": image_id,
                "question_count": 0,
                "questions": [],
            },
        )
        item = QuestionOut.model_validate(question).model_dump(mode="json")
        item["crop_region"] = question.crop_region
        group["questions"].append(item)
        group["question_count"] += 1
    return list(groups.values())


@router.post("/review/images/{image_id}/decisions")
async def decide_image_reviews(
    image_id: str,
    data: ReviewDecisionRequest,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    decisions = {str(item.question_id): item.decision for item in data.decisions}
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
            WrongQuestion.collection_status == "pending_review",
        )
        .with_for_update()
    )
    questions = result.scalars().all()
    if len(questions) != len(decisions):
        raise HTTPException(status_code=404, detail="Review question not found")
    for question in questions:
        question.collection_status = (
            "collected" if decisions[str(question.id)] == "collect" else "ignored"
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
        "collected": sum(item == "collect" for item in decisions.values()),
        "ignored": sum(item == "ignore" for item in decisions.values()),
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
    if not questions:
        raise HTTPException(status_code=409, detail="No pending review questions")
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
        image.status = "needs_review"
        image.recognition_correction = None
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
