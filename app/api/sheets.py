from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_student
from app.models.wrong_question import WrongQuestion
from app.models.practice_sheet import PracticeSheet
from app.models.sheet_item import SheetItem
from app.models.student import Student
from app.schemas.sheet import SheetCreate, SheetOut
from app.services.llm import generate_derivative
from app.services.derivative import generate_derivative_rule
from app.services.pdf import generate_sheet_pdf
from app.config import settings

router = APIRouter(prefix="/sheets", tags=["sheets"])


@router.post("", response_model=SheetOut)
async def create_sheet(
    data: SheetCreate,
    student_id: str = Depends(get_current_student),
    db: AsyncSession = Depends(get_db),
):
    # 获取学生信息
    s = await db.execute(select(Student).where(Student.id == student_id))
    student = s.scalar_one()

    # 获取选中的错题
    qs = await db.execute(
        select(WrongQuestion).where(
            WrongQuestion.id.in_(data.question_ids),
            WrongQuestion.student_id == student_id,
        )
    )
    questions = qs.scalars().all()
    if not questions:
        raise HTTPException(status_code=400, detail="No valid questions selected")

    subject = questions[0].subject or "math"

    # 创建 practice_sheet
    sheet = PracticeSheet(
        student_id=student_id,
        title=data.title,
        config_json=data.model_dump(),
    )
    db.add(sheet)
    await db.flush()

    # 生成 sheet_items（原题 + 衍生题）
    sort = 0
    for q in questions:
        # 原题
        db.add(SheetItem(
            sheet_id=sheet.id,
            wrong_question_id=q.id,
            question_type="original",
            question_text=q.ocr_text or "",
            sort_order=sort,
            generation_method="ocr",
        ))
        sort += 1

        # 衍生题
        target_diff = min(5, (q.difficulty or 2) + data.difficulty_boost)
        method = "rule"
        derived_text = q.ocr_text or ""

        if settings.LLM_API_KEY:
            try:
                derived_text = await generate_derivative(
                    question_text=q.ocr_text or "",
                    problem_schema=q.problem_schema or {},
                    difficulty=q.difficulty or 2,
                    target_difficulty=target_diff,
                    subject=q.subject or "math",
                )
                method = "llm"
            except Exception:
                derived_text = generate_derivative_rule(
                    q.ocr_text or "", q.problem_schema or {}, target_diff, q.subject or "math"
                )
        else:
            derived_text = generate_derivative_rule(
                q.ocr_text or "", q.problem_schema or {}, target_diff, q.subject or "math"
            )

        db.add(SheetItem(
            sheet_id=sheet.id,
            wrong_question_id=q.id,
            question_type="derived",
            derived_from=q.id,
            question_text=derived_text,
            sort_order=sort,
            generation_method=method,
        ))
        sort += 1

    await db.commit()
    await db.refresh(sheet)

    # 从 sheet_items 提取原题和衍生题文本
    items_result = await db.execute(
        select(SheetItem).where(SheetItem.sheet_id == sheet.id).order_by(SheetItem.sort_order)
    )
    items = items_result.scalars().all()
    originals = [it.question_text for it in items if it.question_type == "original"]
    deriveds = [it.question_text for it in items if it.question_type == "derived"]

    pdf_url = generate_sheet_pdf(
        student_name=student.nickname or "学生",
        subject=subject,
        title=data.title,
        original_items=originals,
        derived_items=deriveds,
    )
    sheet.pdf_url = pdf_url
    await db.commit()
    await db.refresh(sheet)

    return sheet


@router.get("", response_model=list[SheetOut])
async def list_sheets(
    student_id: str = Depends(get_current_student),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(PracticeSheet)
        .where(PracticeSheet.student_id == student_id)
        .order_by(PracticeSheet.created_at.desc())
        .limit(20)
    )
    return result.scalars().all()
