import uuid, os, aiofiles
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.api.deps import get_default_student
from app.models.wrong_image import WrongImage
from app.models.student import Student
from app.tasks.process_image import process_image
from app.config import settings

router = APIRouter(prefix="/upload", tags=["upload"])


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
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    image = None
    persisted = False
    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(await file.read())

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
