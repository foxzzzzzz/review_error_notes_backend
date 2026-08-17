from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_default_student
from app.config import settings
from app.database import get_db
from app.models.practice_sheet import PracticeSheet
from app.models.practice_attempt import PracticeAttempt
from app.models.practice_result import PracticeResult
from app.models.sheet_item import SheetItem
from app.models.student import Student
from app.models.wrong_question import WrongQuestion
from app.schemas.sheet import (
    AttemptCreate,
    AttemptOut,
    AttemptUpdate,
    SheetCreate,
    SheetGenerationOut,
    SheetOut,
    SheetReviewOut,
)
from app.services.practice_question import (
    MissingPracticePromptError,
    build_printable_questions,
    order_questions,
)
from app.services.practice_results import (
    PracticeResultValidationError,
    apply_group_result,
    build_review_groups,
    calculate_attempt_summary,
    group_item_results,
)

router = APIRouter(prefix="/sheets", tags=["sheets"])


def enqueue_sheet_generation(sheet_id: str) -> None:
    from app.tasks.generate_sheet import generate_sheet_task

    generate_sheet_task.delay(sheet_id)


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


async def _owned_sheet(db, student_id, sheet_id, *, lock=False):
    statement = select(PracticeSheet).where(
        PracticeSheet.id == sheet_id,
        PracticeSheet.student_id == student_id,
    )
    if lock:
        statement = statement.with_for_update()
    sheet = await db.scalar(statement)
    if sheet is None:
        raise HTTPException(status_code=404, detail="Sheet not found")
    return sheet


async def _sheet_items(db, sheet_id):
    result = await db.execute(
        select(SheetItem)
        .where(SheetItem.sheet_id == sheet_id)
        .order_by(SheetItem.sort_order, SheetItem.id)
    )
    return result.scalars().all()


async def _attempt_out(db, attempt):
    return (await _attempts_out(db, [attempt]))[0]


async def _attempts_out(db, attempts):
    attempt_rows = list(attempts)
    if not attempt_rows:
        return []
    result = await db.execute(
        select(PracticeResult)
        .where(
            PracticeResult.attempt_id.in_(
                [attempt.id for attempt in attempt_rows]
            )
        )
        .order_by(PracticeResult.created_at, PracticeResult.id)
    )
    items_by_attempt = {}
    for item in result.scalars().all():
        items_by_attempt.setdefault(item.attempt_id, []).append(item)
    return [
        AttemptOut(
            id=attempt.id,
            sheet_id=attempt.sheet_id,
            attempt_no=attempt.attempt_no,
            correct_count=attempt.correct_count,
            total_count=attempt.total_count,
            accuracy=float(attempt.accuracy),
            completed_at=attempt.completed_at,
            updated_at=attempt.updated_at,
            items=[
                {
                    "sheet_item_id": item.sheet_item_id,
                    "is_correct": item.is_correct,
                }
                for item in items_by_attempt.get(attempt.id, [])
            ],
        )
        for attempt in attempt_rows
    ]


async def _latest_attempt_ids_by_question(db, question_ids):
    if not question_ids:
        return {}
    result = await db.execute(
        select(
            PracticeResult.wrong_question_id,
            PracticeAttempt.id,
        )
        .join(
            PracticeAttempt,
            PracticeAttempt.id == PracticeResult.attempt_id,
        )
        .where(PracticeResult.wrong_question_id.in_(question_ids))
        .order_by(
            PracticeResult.wrong_question_id,
            PracticeAttempt.created_at.desc(),
            PracticeAttempt.id.desc(),
        )
    )
    latest = {}
    for question_id, attempt_id in result.all():
        latest.setdefault(question_id, attempt_id)
    return latest


def _attempt_conflict():
    return HTTPException(
        status_code=409,
        detail={
            "code": "attempt_conflict",
            "message": "练习结果已发生变化，请刷新后重试",
        },
    )


def _require_completed(sheet) -> None:
    if sheet.generation_status == "completed":
        return
    detail = (
        sheet.generation_error_message
        if sheet.generation_status == "failed" and sheet.generation_error_message
        else "错题集尚未生成完成"
    )
    raise HTTPException(status_code=409, detail=detail)


@router.post("", response_model=SheetOut, status_code=status.HTTP_202_ACCEPTED)
async def create_sheet(
    data: SheetCreate,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    if not data.question_ids:
        raise HTTPException(status_code=400, detail="No valid questions selected")
    if len(set(data.question_ids)) != len(data.question_ids):
        raise HTTPException(status_code=400, detail="Duplicate question IDs are not allowed")

    question_result = await db.execute(
        select(WrongQuestion).where(
            WrongQuestion.id.in_(data.question_ids),
            WrongQuestion.student_id == student.id,
            WrongQuestion.deleted_at.is_(None),
            WrongQuestion.collection_status == "collected",
        )
    )
    questions = order_questions(question_result.scalars().all(), data.question_ids)
    if len(questions) != len(data.question_ids):
        raise HTTPException(status_code=400, detail="Some selected questions are unavailable")

    try:
        build_printable_questions(questions)
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

    sheet = PracticeSheet(
        student_id=student.id,
        title=data.title,
        config_json=data.model_dump(),
        generation_status="pending",
        generation_total=len(questions),
        generation_completed=0,
    )
    db.add(sheet)
    await db.commit()
    await db.refresh(sheet)
    try:
        enqueue_sheet_generation(str(sheet.id))
    except Exception as exc:
        sheet.generation_status = "failed"
        sheet.generation_error_code = "sheet_queue_unavailable"
        sheet.generation_error_message = "错题集生成服务暂时不可用，请稍后重试"
        await db.commit()
        raise HTTPException(
            status_code=503,
            detail=sheet.generation_error_message,
        ) from exc
    return sheet


@router.get("", response_model=list[SheetOut])
async def list_sheets(
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    attempt_count = (
        select(func.count(PracticeAttempt.id))
        .where(PracticeAttempt.sheet_id == PracticeSheet.id)
        .correlate(PracticeSheet)
        .scalar_subquery()
    )
    latest_accuracy = (
        select(PracticeAttempt.accuracy)
        .where(PracticeAttempt.sheet_id == PracticeSheet.id)
        .order_by(PracticeAttempt.attempt_no.desc())
        .limit(1)
        .correlate(PracticeSheet)
        .scalar_subquery()
    )
    result = await db.execute(
        select(PracticeSheet, attempt_count, latest_accuracy)
        .where(PracticeSheet.student_id == student.id)
        .order_by(PracticeSheet.created_at.desc())
        .limit(20)
    )
    return [
        SheetOut(
            id=sheet.id,
            title=sheet.title,
            config_json=sheet.config_json,
            pdf_url=sheet.pdf_url,
            created_at=sheet.created_at,
            updated_at=sheet.updated_at,
            generation_status=sheet.generation_status,
            generation_total=sheet.generation_total,
            generation_completed=sheet.generation_completed,
            generation_error_code=sheet.generation_error_code,
            generation_error_message=sheet.generation_error_message,
            items=[],
            latest_accuracy=(
                float(row_latest_accuracy)
                if row_latest_accuracy is not None
                else None
            ),
            attempt_count=row_attempt_count,
            practice_status=(
                "completed" if row_attempt_count else "unpracticed"
            ),
        )
        for sheet, row_attempt_count, row_latest_accuracy in result.all()
    ]


@router.get("/{sheet_id}/generation", response_model=SheetGenerationOut)
async def get_sheet_generation(
    sheet_id: str,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    return await _owned_sheet(db, student.id, sheet_id)


@router.post("/{sheet_id}/retry", response_model=SheetGenerationOut)
async def retry_sheet_generation(
    sheet_id: str,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    sheet = await _owned_sheet(db, student.id, sheet_id, lock=True)
    if sheet.generation_status != "failed":
        raise HTTPException(status_code=409, detail="当前错题集不能重新生成")
    sheet.generation_status = "pending"
    sheet.generation_completed = 0
    sheet.generation_error_code = None
    sheet.generation_error_message = None
    await db.commit()
    await db.refresh(sheet)
    try:
        enqueue_sheet_generation(str(sheet.id))
    except Exception as exc:
        sheet.generation_status = "failed"
        sheet.generation_error_code = "sheet_queue_unavailable"
        sheet.generation_error_message = "错题集生成服务暂时不可用，请稍后重试"
        await db.commit()
        raise HTTPException(
            status_code=503,
            detail=sheet.generation_error_message,
        ) from exc
    return sheet


@router.get("/{sheet_id}/review", response_model=SheetReviewOut)
async def get_sheet_review(
    sheet_id: str,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    sheet = await _owned_sheet(db, student.id, sheet_id)
    _require_completed(sheet)
    items = await _sheet_items(db, sheet.id)
    latest_attempt = await db.scalar(
        select(PracticeAttempt)
        .where(
            PracticeAttempt.sheet_id == sheet.id,
            PracticeAttempt.student_id == student.id,
        )
        .order_by(PracticeAttempt.attempt_no.desc())
        .limit(1)
    )
    result_by_item = {}
    attempt_out = None
    if latest_attempt is not None:
        attempt_out = await _attempt_out(db, latest_attempt)
        result_by_item = {
            item.sheet_item_id: item.is_correct for item in attempt_out.items
        }
    return SheetReviewOut(
        sheet_id=sheet.id,
        title=sheet.title,
        latest_attempt=attempt_out,
        groups=build_review_groups(items, result_by_item),
    )


@router.get("/{sheet_id}/attempts", response_model=list[AttemptOut])
async def list_sheet_attempts(
    sheet_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    sheet = await _owned_sheet(db, student.id, sheet_id)
    _require_completed(sheet)
    result = await db.execute(
        select(PracticeAttempt)
        .where(
            PracticeAttempt.sheet_id == sheet.id,
            PracticeAttempt.student_id == student.id,
        )
        .order_by(PracticeAttempt.attempt_no.desc())
        .limit(limit)
        .offset(offset)
    )
    return await _attempts_out(db, result.scalars().all())


@router.post("/{sheet_id}/attempts", response_model=AttemptOut)
async def create_sheet_attempt(
    sheet_id: str,
    data: AttemptCreate,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.scalar(
        select(PracticeAttempt).where(
            PracticeAttempt.student_id == student.id,
            PracticeAttempt.idempotency_key == data.idempotency_key,
        )
    )
    if existing is not None:
        if str(existing.sheet_id) != str(sheet_id):
            raise _attempt_conflict()
        return await _attempt_out(db, existing)

    sheet = await _owned_sheet(db, student.id, sheet_id, lock=True)
    _require_completed(sheet)
    items = await _sheet_items(db, sheet.id)
    answers = {item.sheet_item_id: item.is_correct for item in data.items}
    try:
        groups = group_item_results(items, answers)
    except PracticeResultValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    summary = calculate_attempt_summary(
        groups,
        correct_count=sum(answers.values()),
        total_count=len(answers),
    )
    latest_no = await db.scalar(
        select(func.max(PracticeAttempt.attempt_no)).where(
            PracticeAttempt.sheet_id == sheet.id
        )
    )
    attempt = PracticeAttempt(
        sheet_id=sheet.id,
        student_id=student.id,
        attempt_no=(latest_no or 0) + 1,
        idempotency_key=data.idempotency_key,
        correct_count=summary.correct_count,
        total_count=summary.total_count,
        accuracy=summary.accuracy,
        completed_at=_utc_naive(data.completed_at),
    )
    db.add(attempt)
    await db.flush()

    group_sizes = {
        group.wrong_question_id: len(group.item_ids) - 1 for group in groups
    }
    for item in items:
        db.add(
            PracticeResult(
                attempt_id=attempt.id,
                sheet_item_id=item.id,
                wrong_question_id=item.wrong_question_id,
                is_correct=answers[item.id],
                derived_count=group_sizes.get(item.wrong_question_id, 0),
                question_snapshot=item.question_snapshot,
            )
        )
    await db.flush()

    question_ids = [
        group.wrong_question_id
        for group in groups
        if group.wrong_question_id is not None
    ]
    question_result = await db.execute(
        select(WrongQuestion)
        .where(
            WrongQuestion.id.in_(question_ids),
            WrongQuestion.student_id == student.id,
            WrongQuestion.deleted_at.is_(None),
        )
        .order_by(WrongQuestion.id)
        .with_for_update()
    )
    questions = {
        question.id: question for question in question_result.scalars().all()
    }
    latest_attempt_ids = await _latest_attempt_ids_by_question(
        db,
        question_ids,
    )
    practiced_at = _utc_naive(data.completed_at)
    for group in groups:
        question = questions.get(group.wrong_question_id)
        if question is not None:
            apply_group_result(
                question,
                all_correct=group.all_correct,
                previous_correct=None,
                now=practiced_at,
                update_mastery=(
                    latest_attempt_ids.get(group.wrong_question_id)
                    == attempt.id
                ),
            )
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        duplicate = await db.scalar(
            select(PracticeAttempt).where(
                PracticeAttempt.student_id == student.id,
                PracticeAttempt.idempotency_key == data.idempotency_key,
            )
        )
        if duplicate is None or duplicate.sheet_id != sheet.id:
            raise
        return await _attempt_out(db, duplicate)
    await db.refresh(attempt)
    return await _attempt_out(db, attempt)


@router.patch(
    "/{sheet_id}/attempts/{attempt_id}",
    response_model=AttemptOut,
)
async def update_sheet_attempt(
    sheet_id: str,
    attempt_id: str,
    data: AttemptUpdate,
    student: Student = Depends(get_default_student),
    db: AsyncSession = Depends(get_db),
):
    sheet = await _owned_sheet(db, student.id, sheet_id, lock=True)
    _require_completed(sheet)
    attempt = await db.scalar(
        select(PracticeAttempt)
        .where(
            PracticeAttempt.id == attempt_id,
            PracticeAttempt.sheet_id == sheet_id,
            PracticeAttempt.student_id == student.id,
        )
        .with_for_update()
    )
    if attempt is None or attempt.updated_at != _utc_naive(data.updated_at):
        raise _attempt_conflict()
    latest_id = await db.scalar(
        select(PracticeAttempt.id)
        .where(PracticeAttempt.sheet_id == sheet_id)
        .order_by(PracticeAttempt.attempt_no.desc())
        .limit(1)
    )
    if latest_id != attempt.id:
        raise _attempt_conflict()

    items = await _sheet_items(db, sheet_id)
    result = await db.execute(
        select(PracticeResult)
        .where(PracticeResult.attempt_id == attempt.id)
        .with_for_update()
    )
    old_results = result.scalars().all()
    old_answers = {
        item.sheet_item_id: item.is_correct for item in old_results
    }
    next_answers = {item.sheet_item_id: item.is_correct for item in data.items}
    try:
        old_groups = group_item_results(items, old_answers)
        next_groups = group_item_results(items, next_answers)
    except PracticeResultValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    old_by_question = {
        group.wrong_question_id: group.all_correct for group in old_groups
    }
    summary = calculate_attempt_summary(
        next_groups,
        correct_count=sum(next_answers.values()),
        total_count=len(next_answers),
    )
    for result_item in old_results:
        result_item.is_correct = next_answers[result_item.sheet_item_id]
    attempt.correct_count = summary.correct_count
    attempt.total_count = summary.total_count
    attempt.accuracy = summary.accuracy

    question_ids = [
        group.wrong_question_id
        for group in next_groups
        if group.wrong_question_id is not None
    ]
    question_result = await db.execute(
        select(WrongQuestion)
        .where(
            WrongQuestion.id.in_(question_ids),
            WrongQuestion.student_id == student.id,
            WrongQuestion.deleted_at.is_(None),
        )
        .order_by(WrongQuestion.id)
        .with_for_update()
    )
    questions = {
        question.id: question for question in question_result.scalars().all()
    }
    latest_attempt_ids = await _latest_attempt_ids_by_question(
        db,
        question_ids,
    )
    practiced_at = attempt.completed_at
    for group in next_groups:
        question = questions.get(group.wrong_question_id)
        if question is not None:
            apply_group_result(
                question,
                all_correct=group.all_correct,
                previous_correct=old_by_question.get(group.wrong_question_id),
                now=practiced_at,
                update_mastery=(
                    latest_attempt_ids.get(group.wrong_question_id)
                    == attempt.id
                ),
            )
    await db.commit()
    await db.refresh(attempt)
    return await _attempt_out(db, attempt)
