import asyncio
from datetime import datetime
from types import SimpleNamespace
from uuid import UUID, uuid4


def _row(
    attempt_id,
    question_id,
    is_correct,
    *,
    created_at,
    completed_at,
):
    from app.services.sheet_deletion import QuestionHistoryRow

    return QuestionHistoryRow(
        attempt_id=UUID(attempt_id),
        question_id=UUID(question_id),
        is_correct=is_correct,
        attempt_created_at=created_at,
        attempt_completed_at=completed_at,
    )


def test_question_history_groups_items_and_uses_latest_attempt_state():
    from app.services.sheet_deletion import summarize_question_history

    question_id = "00000000-0000-0000-0000-000000000010"
    older = datetime(2026, 8, 18, 9, 0, 0)
    newer = datetime(2026, 8, 18, 10, 0, 0)
    rows = [
        _row(
            "00000000-0000-0000-0000-000000000001",
            question_id,
            True,
            created_at=older,
            completed_at=older,
        ),
        _row(
            "00000000-0000-0000-0000-000000000001",
            question_id,
            False,
            created_at=older,
            completed_at=older,
        ),
        _row(
            "00000000-0000-0000-0000-000000000002",
            question_id,
            True,
            created_at=newer,
            completed_at=newer,
        ),
        _row(
            "00000000-0000-0000-0000-000000000002",
            question_id,
            True,
            created_at=newer,
            completed_at=newer,
        ),
    ]

    state = summarize_question_history(rows)[UUID(question_id)]

    assert state.incorrect_attempt_count == 1
    assert state.latest_all_correct is True
    assert state.latest_completed_at == newer


def test_question_history_breaks_created_at_ties_by_attempt_id():
    from app.services.sheet_deletion import summarize_question_history

    question_id = "00000000-0000-0000-0000-000000000010"
    created_at = datetime(2026, 8, 18, 10, 0, 0)
    rows = [
        _row(
            "00000000-0000-0000-0000-000000000001",
            question_id,
            True,
            created_at=created_at,
            completed_at=datetime(2026, 8, 18, 11, 0, 0),
        ),
        _row(
            "00000000-0000-0000-0000-000000000002",
            question_id,
            False,
            created_at=created_at,
            completed_at=datetime(2026, 8, 18, 9, 0, 0),
        ),
    ]

    state = summarize_question_history(rows)[UUID(question_id)]

    assert state.latest_all_correct is False
    assert state.latest_completed_at == datetime(2026, 8, 18, 9, 0, 0)


def test_apply_deleted_sheet_state_subtracts_wrong_attempts_and_uses_remaining_latest():
    from app.services.sheet_deletion import (
        RemainingQuestionState,
        apply_deleted_sheet_state,
    )

    latest_at = datetime(2026, 8, 18, 10, 0, 0)
    question = SimpleNamespace(
        wrong_count=4,
        mastery_status="learning",
        mastered_at=None,
        last_practiced_at=datetime(2026, 8, 18, 11, 0, 0),
    )
    remaining = RemainingQuestionState(
        incorrect_attempt_count=1,
        latest_all_correct=True,
        latest_completed_at=latest_at,
    )

    apply_deleted_sheet_state(
        question,
        deleted_incorrect_attempt_count=2,
        remaining_state=remaining,
    )

    assert question.wrong_count == 2
    assert question.mastery_status == "mastered"
    assert question.mastered_at == latest_at
    assert question.last_practiced_at == latest_at


def test_apply_deleted_sheet_state_resets_mastery_without_remaining_history():
    from app.services.sheet_deletion import apply_deleted_sheet_state

    question = SimpleNamespace(
        wrong_count=2,
        mastery_status="mastered",
        mastered_at=datetime(2026, 8, 18, 10, 0, 0),
        last_practiced_at=datetime(2026, 8, 18, 10, 0, 0),
    )

    apply_deleted_sheet_state(
        question,
        deleted_incorrect_attempt_count=5,
        remaining_state=None,
    )

    assert question.wrong_count == 1
    assert question.mastery_status == "learning"
    assert question.mastered_at is None
    assert question.last_practiced_at is None


class _DeleteSheetDb:
    def __init__(self):
        self.executed = []
        self.added = []
        self.flush_calls = 0

    async def execute(self, statement):
        self.executed.append(statement)

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        self.flush_calls += 1
        for value in self.added:
            if value.id is None:
                value.id = uuid4()


def test_delete_sheet_data_removes_dependents_and_reconciles_questions(monkeypatch):
    from app.services import sheet_deletion

    sheet_id = uuid4()
    question_id = uuid4()
    deleted_attempt_id = uuid4()
    remaining_attempt_id = uuid4()
    deleted_at = datetime(2026, 8, 18, 9, 0, 0)
    remaining_at = datetime(2026, 8, 18, 10, 0, 0)
    target_rows = [
        sheet_deletion.QuestionHistoryRow(
            attempt_id=deleted_attempt_id,
            question_id=question_id,
            is_correct=True,
            attempt_created_at=deleted_at,
            attempt_completed_at=deleted_at,
        ),
        sheet_deletion.QuestionHistoryRow(
            attempt_id=deleted_attempt_id,
            question_id=question_id,
            is_correct=False,
            attempt_created_at=deleted_at,
            attempt_completed_at=deleted_at,
        ),
    ]
    remaining_rows = [
        sheet_deletion.QuestionHistoryRow(
            attempt_id=remaining_attempt_id,
            question_id=question_id,
            is_correct=True,
            attempt_created_at=remaining_at,
            attempt_completed_at=remaining_at,
        )
    ]
    question = SimpleNamespace(
        id=question_id,
        wrong_count=3,
        mastery_status="learning",
        mastered_at=None,
        last_practiced_at=deleted_at,
    )
    concurrency_order = []

    async def load_history(_db, *, sheet_id=None, question_ids=None, excluded_sheet_id=None):
        if sheet_id is not None:
            concurrency_order.append("target_history")
            return target_rows
        assert question_ids == {question_id}
        assert excluded_sheet_id == sheet_id_value
        concurrency_order.append("remaining_history")
        return remaining_rows

    async def lock_questions(_db, student_id, question_ids):
        assert student_id == "student-id"
        assert question_ids == {question_id}
        concurrency_order.append("question_lock")
        return [question]

    sheet_id_value = sheet_id
    monkeypatch.setattr(sheet_deletion, "_load_history_rows", load_history)
    monkeypatch.setattr(sheet_deletion, "_lock_questions", lock_questions)
    db = _DeleteSheetDb()
    sheet = SimpleNamespace(id=sheet_id, pdf_url="/pdfs/sheet.pdf")

    cleanup_job_id = asyncio.run(
        sheet_deletion.delete_sheet_data(db, sheet, "student-id")
    )

    assert [statement.table.name for statement in db.executed] == [
        "practice_results",
        "practice_attempts",
        "sheet_items",
        "practice_sheets",
    ]
    assert question.wrong_count == 2
    assert question.mastery_status == "mastered"
    assert question.mastered_at == remaining_at
    assert question.last_practiced_at == remaining_at
    assert len(db.added) == 1
    assert db.added[0].storage_kind == "pdf"
    assert cleanup_job_id == db.added[0].id
    assert db.flush_calls == 1
    assert concurrency_order == [
        "target_history",
        "question_lock",
        "remaining_history",
    ]


def test_delete_sheet_data_without_practice_or_pdf_only_deletes_sheet_rows(monkeypatch):
    from app.services import sheet_deletion

    async def load_history(_db, **_kwargs):
        return []

    async def lock_questions(_db, _student_id, _question_ids):
        raise AssertionError("questions must not be locked without practice history")

    monkeypatch.setattr(sheet_deletion, "_load_history_rows", load_history)
    monkeypatch.setattr(sheet_deletion, "_lock_questions", lock_questions)
    db = _DeleteSheetDb()
    sheet = SimpleNamespace(id=uuid4(), pdf_url=None)

    cleanup_job_id = asyncio.run(
        sheet_deletion.delete_sheet_data(db, sheet, "student-id")
    )

    assert [statement.table.name for statement in db.executed] == [
        "practice_results",
        "practice_attempts",
        "sheet_items",
        "practice_sheets",
    ]
    assert cleanup_job_id is None
    assert db.added == []
    assert db.flush_calls == 0


def test_question_lock_includes_soft_deleted_questions_for_reconciliation():
    from sqlalchemy.dialects import postgresql

    from app.services.sheet_deletion import _lock_questions

    class LockDb:
        statement = None

        async def execute(self, statement):
            self.statement = statement
            return SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: []),
            )

    db = LockDb()
    asyncio.run(_lock_questions(db, uuid4(), {uuid4()}))

    sql = str(db.statement.compile(dialect=postgresql.dialect()))
    assert "wrong_questions.deleted_at IS NULL" not in sql
