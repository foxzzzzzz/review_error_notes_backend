import uuid, os, aiofiles
from io import BytesIO
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.api.deps import get_default_student
from app.models.wrong_image import WrongImage
from app.models.student import Student
from app.tasks.process_image import process_image
from app.config import settings
from PIL import Image, UnidentifiedImageError

router = APIRouter(prefix="/upload", tags=["upload"])


def _validated_image_upload(file_data: bytes) -> str:
    try:
        with Image.open(BytesIO(file_data)) as image:
            image.verify()
            image_format = image.format
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="上传文件不是有效图片") from exc
    extensions = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    if image_format not in extensions:
        raise HTTPException(status_code=422, detail="仅支持 JPEG、PNG 或 WebP 图片")
    return extensions[image_format]


def serialize_image_status(image: WrongImage) -> dict:
    return {
        "image_id": str(image.id),
        "status": image.status,
        "question_count": image.question_count,
        "error_code": image.error_code,
        "error_message": image.error_message,
    }


@router.get("/images/status")
async def get_image_statuses(
    image_ids: list[str] = Query(..., min_length=1, max_length=9),
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(WrongImage).where(
            WrongImage.student_id == student.id,
            WrongImage.id.in_(image_ids),
        )
    )
    return [serialize_image_status(image) for image in result.scalars().all()]


@router.post("/images/{image_id}/retry")
async def retry_image(
    image_id: str,
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
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if image.status != "failed":
        raise HTTPException(status_code=409, detail="Only failed images can be retried")

    image.status = "pending"
    image.error_code = None
    image.error_message = None
    await db.commit()
    try:
        filepath = str(Path(settings.UPLOAD_DIR) / Path(image.original_url).name)
        process_image.delay(str(image.id), filepath)
    except Exception:
        image.status = "failed"
        image.error_code = "task_dispatch_failed"
        image.error_message = "处理任务投递失败，请重试"
        await db.commit()
        raise HTTPException(status_code=503, detail=image.error_message)
    return {"image_id": str(image.id), "status": "pending"}


@router.post("/image")
async def upload_image(
    file: UploadFile = File(...),
    subject: Literal["math", "chinese", "english"] | None = Form(None),
    grade: int | None = Form(None, ge=1, le=6),
    semester: int | None = Form(None, ge=1, le=2),
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    resolved_grade = grade if grade is not None else student.grade
    resolved_semester = semester if semester is not None else student.semester
    if resolved_grade is None or resolved_semester is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "student_profile_required",
                "message": "请先选择年级和册别",
            },
        )

    # 保存文件
    file_data = await file.read(settings.UPLOAD_MAX_BYTES + 1)
    if len(file_data) > settings.UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="图片文件过大")
    ext = _validated_image_upload(file_data)
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    image = None
    persisted = False
    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(file_data)

        # 获取学生默认年级/册别
        # 创建 wrong_image 记录
        image = WrongImage(
            student_id=student.id,
            original_url=f"/uploads/{filename}",
            subject=subject,
            grade=resolved_grade,
            semester=resolved_semester,
            status="pending",
        )
        db.add(image)
        await db.commit()
        persisted = True
        await db.refresh(image)

        # 投递异步 OCR 任务
        process_image.delay(str(image.id), filepath)
    except Exception:
        if persisted and image is not None:
            try:
                await db.delete(image)
                await db.commit()
            except Exception:
                await db.rollback()
        else:
            await db.rollback()
        Path(filepath).unlink(missing_ok=True)
        raise

    return {"image_id": str(image.id), "status": "pending"}
