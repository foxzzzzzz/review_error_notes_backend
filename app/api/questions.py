from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.api.deps import get_current_student
from app.models.wrong_question import WrongQuestion
from app.schemas.question import QuestionOut, QuestionUpdate

router = APIRouter(prefix="/questions", tags=["questions"])

@router.get("", response_model=list[QuestionOut])
async def list_questions(
    subject: str = None,
    grade: int = None,
    semester: int = None,
    status: str = None,
    tag: str = None,
    limit: int = Query(20, le=100),
    offset: int = 0,
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
    q = q.order_by(WrongQuestion.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{question_id}", response_model=QuestionOut)
async def get_question(question_id: str, student_id=Depends(get_current_student), db=Depends(get_db)):
    result = await db.execute(
        select(WrongQuestion).where(
            WrongQuestion.id == question_id,
            WrongQuestion.student_id == student_id,
        )
    )
    return result.scalar_one()


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
    q = result.scalar_one()
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(q, k, v)
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
    q = result.scalar_one()
    await db.delete(q)
    await db.commit()
    return {"ok": True}
