from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_student
from app.config import settings
from app.database import get_db
from app.models.practice_sheet import PracticeSheet
from app.models.sheet_item import SheetItem
from app.models.student import Student
from app.models.wrong_question import WrongQuestion
from app.schemas.sheet import SheetCreate, SheetOut
from app.services.derivative import (
    DerivativeGenerationError,
    generate_derivative_variants,
)
from app.services.pdf import generate_sheet_pdf
from app.services.pdf_storage import remove_generated_pdf
from app.services.practice_question import (
    MissingPracticePromptError,
    build_printable_questions,
    order_questions,
)

router = APIRouter(prefix="/sheets", tags=["sheets"])


@router.post("", response_model=SheetOut)
async def create_sheet(
    data: SheetCreate,
    student_id: str = Depends(get_current_student),
    db: AsyncSession = Depends(get_db),
):
    if not data.question_ids:
        raise HTTPException(status_code=400, detail="No valid questions selected")
    if len(set(data.question_ids)) != len(data.question_ids):
        raise HTTPException(status_code=400, detail="Duplicate question IDs are not allowed")

    student_result = await db.execute(select(Student).where(Student.id == student_id))
    student = student_result.scalar_one()
    question_result = await db.execute(
        select(WrongQuestion).where(
            WrongQuestion.id.in_(data.question_ids),
            WrongQuestion.student_id == student_id,
        )
    )
    questions = order_questions(question_result.scalars().all(), data.question_ids)
    if len(questions) != len(data.question_ids):
        raise HTTPException(status_code=400, detail="Some selected questions are unavailable")

    try:
        originals = build_printable_questions(questions)
    except MissingPracticePromptError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"有 {exc.count} 道错题缺少结构化题干，"
                "请重新上传图片识别后再出卷"
            ),
        ) from exc

    if data.derived_per_original > 0 and not settings.LLM_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="衍生题服务未配置，请选择“仅原题”或联系管理员",
        )

    groups = []
    try:
        for question, original in zip(questions, originals):
            target_difficulty = min(
                5,
                (question.difficulty or 2) + data.difficulty_boost,
            )
            derivatives = await generate_derivative_variants(
                original=original,
                difficulty=question.difficulty or 2,
                target_difficulty=target_difficulty,
                subject=question.subject or "math",
                count=data.derived_per_original,
            )
            groups.append(
                {
                    "original": original.model_dump(exclude={"answer"}),
                    "derivatives": [
                        item.model_dump(exclude={"answer"}) for item in derivatives
                    ],
                }
            )
    except DerivativeGenerationError as exc:
        raise HTTPException(
            status_code=502,
            detail="衍生题生成失败，请稍后重试或选择“仅原题”",
        ) from exc

    subject = questions[0].subject or "math"
    sheet = PracticeSheet(
        student_id=student_id,
        title=data.title,
        config_json=data.model_dump(),
    )
    db.add(sheet)

    pdf_url = None
    try:
        await db.flush()
        sort_order = 0
        for question, group in zip(questions, groups):
            original = group["original"]
            db.add(
                SheetItem(
                    sheet_id=sheet.id,
                    wrong_question_id=question.id,
                    question_type="original",
                    question_text=original["display_text"],
                    sort_order=sort_order,
                    generation_method="vision",
                )
            )
            sort_order += 1
            for derivative in group["derivatives"]:
                db.add(
                    SheetItem(
                        sheet_id=sheet.id,
                        wrong_question_id=question.id,
                        question_type="derived",
                        derived_from=question.id,
                        question_text=derivative["display_text"],
                        sort_order=sort_order,
                        generation_method="llm",
                    )
                )
                sort_order += 1

        await db.flush()
        pdf_url = generate_sheet_pdf(
            student_name=student.nickname or "学生",
            subject=subject,
            title=data.title,
            groups=groups,
        )
        sheet.pdf_url = pdf_url
        await db.commit()
    except Exception:
        await db.rollback()
        if pdf_url:
            remove_generated_pdf(pdf_url)
        raise

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
