import uuid, os, aiofiles
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.api.deps import get_current_student
from app.models.wrong_image import WrongImage
from app.models.student import Student
from app.tasks.process_image import process_image
from sqlalchemy import select
from app.config import settings

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/image")
async def upload_image(
    file: UploadFile = File(...),
    student_id: str = Depends(get_current_student),
    db: AsyncSession = Depends(get_db),
):
    # 保存文件
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    async with aiofiles.open(filepath, "wb") as f:
        await f.write(await file.read())

    # 获取学生默认年级/册别
    result = await db.execute(select(Student).where(Student.id == student_id))
    student = result.scalar_one()

    # 创建 wrong_image 记录
    image = WrongImage(
        student_id=student_id,
        original_url=f"/uploads/{filename}",
        grade=student.grade,
        semester=student.semester,
        status="pending",
    )
    db.add(image)
    await db.commit()
    await db.refresh(image)

    # 投递异步 OCR 任务
    process_image.delay(str(image.id), filepath)

    return {"image_id": str(image.id), "status": "pending"}
