"""Delete practice sheets and reconcile affected question state."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, select

from app.models.practice_attempt import PracticeAttempt
from app.models.practice_result import PracticeResult
from app.models.practice_sheet import PracticeSheet
from app.models.sheet_item import SheetItem
from app.models.wrong_question import WrongQuestion
from app.services.file_cleanup import build_file_cleanup_job


@dataclass(frozen=True)
class QuestionHistoryRow:
    attempt_id: UUID
    question_id: UUID
    is_correct: bool
    attempt_created_at: datetime
    attempt_completed_at: datetime


@dataclass(frozen=True)
class RemainingQuestionState:
    incorrect_attempt_count: int
    latest_all_correct: bool
    latest_completed_at: datetime


def summarize_question_history(
    rows: list[QuestionHistoryRow],
) -> dict[UUID, RemainingQuestionState]:
    groups = {}
    for row in rows:
        key = (row.question_id, row.attempt_id)
        group = groups.setdefault(
            key,
            {
                "all_correct": True,
                "created_at": row.attempt_created_at,
                "completed_at": row.attempt_completed_at,
            },
        )
        group["all_correct"] = group["all_correct"] and row.is_correct

    grouped_by_question = {}
    for (question_id, attempt_id), group in groups.items():
        grouped_by_question.setdefault(question_id, []).append(
            (attempt_id, group)
        )

    summaries = {}
    for question_id, attempts in grouped_by_question.items():
        latest_attempt_id, latest = max(
            attempts,
            key=lambda item: (item[1]["created_at"], item[0]),
        )
        del latest_attempt_id
        summaries[question_id] = RemainingQuestionState(
            incorrect_attempt_count=sum(
                not attempt["all_correct"] for _, attempt in attempts
            ),
            latest_all_correct=latest["all_correct"],
            latest_completed_at=latest["completed_at"],
        )
    return summaries


def apply_deleted_sheet_state(
    question,
    *,
    deleted_incorrect_attempt_count: int,
    remaining_state: RemainingQuestionState | None,
) -> None:
    question.wrong_count = max(
        1,
        (question.wrong_count or 1) - deleted_incorrect_attempt_count,
    )
    if remaining_state is None:
        question.mastery_status = "learning"
        question.mastered_at = None
        question.last_practiced_at = None
        return

    question.last_practiced_at = remaining_state.latest_completed_at
    if remaining_state.latest_all_correct:
        question.mastery_status = "mastered"
        question.mastered_at = remaining_state.latest_completed_at
    else:
        question.mastery_status = "learning"
        question.mastered_at = None


async def _load_history_rows(
    db,
    *,
    sheet_id=None,
    question_ids=None,
    excluded_sheet_id=None,
) -> list[QuestionHistoryRow]:
    statement = (
        select(
            PracticeResult.attempt_id,
            PracticeResult.wrong_question_id,
            PracticeResult.is_correct,
            PracticeAttempt.created_at,
            PracticeAttempt.completed_at,
        )
        .join(
            PracticeAttempt,
            PracticeAttempt.id == PracticeResult.attempt_id,
        )
        .where(PracticeResult.wrong_question_id.is_not(None))
    )
    if sheet_id is not None:
        statement = statement.where(PracticeAttempt.sheet_id == sheet_id)
    if question_ids:
        statement = statement.where(
            PracticeResult.wrong_question_id.in_(question_ids)
        )
    if excluded_sheet_id is not None:
        statement = statement.where(PracticeAttempt.sheet_id != excluded_sheet_id)

    result = await db.execute(statement)
    return [
        QuestionHistoryRow(
            attempt_id=attempt_id,
            question_id=question_id,
            is_correct=is_correct,
            attempt_created_at=created_at,
            attempt_completed_at=completed_at,
        )
        for attempt_id, question_id, is_correct, created_at, completed_at in result.all()
    ]


async def _lock_questions(db, student_id, question_ids):
    result = await db.execute(
        select(WrongQuestion)
        .where(
            WrongQuestion.id.in_(question_ids),
            WrongQuestion.student_id == student_id,
        )
        .order_by(WrongQuestion.id)
        .with_for_update()
    )
    return result.scalars().all()


async def delete_sheet_data(db, sheet, student_id) -> UUID | None:
    target_rows = await _load_history_rows(db, sheet_id=sheet.id)
    target_states = summarize_question_history(target_rows)
    question_ids = set(target_states)

    remaining_states = {}
    questions = []
    if question_ids:
        questions = await _lock_questions(db, student_id, question_ids)
        remaining_rows = await _load_history_rows(
            db,
            question_ids=question_ids,
            excluded_sheet_id=sheet.id,
        )
        remaining_states = summarize_question_history(remaining_rows)

    for question in questions:
        apply_deleted_sheet_state(
            question,
            deleted_incorrect_attempt_count=(
                target_states[question.id].incorrect_attempt_count
            ),
            remaining_state=remaining_states.get(question.id),
        )

    attempt_ids = select(PracticeAttempt.id).where(
        PracticeAttempt.sheet_id == sheet.id
    )
    await db.execute(
        delete(PracticeResult).where(PracticeResult.attempt_id.in_(attempt_ids))
    )
    await db.execute(
        delete(PracticeAttempt).where(PracticeAttempt.sheet_id == sheet.id)
    )
    await db.execute(delete(SheetItem).where(SheetItem.sheet_id == sheet.id))
    await db.execute(delete(PracticeSheet).where(PracticeSheet.id == sheet.id))

    cleanup_job = build_file_cleanup_job("pdf", sheet.pdf_url)
    if cleanup_job is None:
        return None
    db.add(cleanup_job)
    await db.flush()
    return cleanup_job.id
