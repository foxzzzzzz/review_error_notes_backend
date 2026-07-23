from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Query, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.config import settings
from app.database import get_db
from app.api.deps import get_current_student
from app.models.wrong_question import WrongQuestion
from app.models.wrong_image import WrongImage
from app.schemas.question import QuestionOut, QuestionUpdate
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
    tag: str = None,
    limit: int = Query(20, le=100),
    offset: int = 0,
    created_from: datetime = None,
    student_id: str = Depends(get_current_student),
    db: AsyncSession = Depends(get_db),
):
    q = select(WrongQuestion).where(WrongQuestion.student_id == student_id)
    if subject:
        q = q.where(WrongQuestion.subject == subject)
    if grade:
        q = q.where(WrongQuestion.grade == grade)
    if semester:
        q = q.where(WrongQuestion.semester == semester)
    if status:
        q = q.where(WrongQuestion.status == status)
    if tag:
        q = q.where(WrongQuestion.tags.any(tag))
    if created_from:
        q = q.where(WrongQuestion.created_at >= _normalize_created_from(created_from))
    q = q.order_by(WrongQuestion.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{question_id}", response_model=QuestionOut)
async def get_question(question_id: str, student_id=Depends(get_current_student), db=Depends(get_db)):
    result = await db.execute(
        select(WrongQuestion, WrongImage)
        .join(WrongImage, WrongImage.id == WrongQuestion.image_id)
        .where(
            WrongQuestion.id == question_id,
            WrongQuestion.student_id == student_id,
        )
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    q, image = row
    data = QuestionOut.model_validate(q).model_dump()
    data.update({
        "crop_region": q.crop_region,
        "image_url": image.original_url,
    })
    return data


@router.get("/{question_id}/image")
async def get_question_image(
    question_id: str,
    view: Literal["crop", "original"] = "crop",
    student_id=Depends(get_current_student),
    db=Depends(get_db),
):
    result = await db.execute(
        select(WrongQuestion, WrongImage)
        .join(WrongImage, WrongImage.id == WrongQuestion.image_id)
        .where(
            WrongQuestion.id == question_id,
            WrongQuestion.student_id == student_id,
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
    student_id=Depends(get_current_student),
    db=Depends(get_db),
):
    result = await db.execute(
        select(WrongQuestion).where(
            WrongQuestion.id == question_id,
            WrongQuestion.student_id == student_id,
        )
    )
    q = result.scalar_one_or_none()
    if not q:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    was_needs_review = q.status == "needs_review"
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
        q.status = "confirmed"
        await db.flush()
        remaining_review = await db.scalar(
            select(WrongQuestion.id)
            .where(
                WrongQuestion.image_id == q.image_id,
                WrongQuestion.status == "needs_review",
            )
            .limit(1)
        )
        if not remaining_review and image:
            image.status = "confirmed"
    await db.commit()
    return {"ok": True}


@router.delete("/{question_id}")
async def delete_question(question_id: str, student_id=Depends(get_current_student), db=Depends(get_db)):
    result = await db.execute(
        select(WrongQuestion).where(
            WrongQuestion.id == question_id,
            WrongQuestion.student_id == student_id,
        )
    )
    q = result.scalar_one_or_none()
    if not q:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    await db.delete(q)
    await db.commit()
    return {"ok": True}
