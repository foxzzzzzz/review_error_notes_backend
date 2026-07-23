import asyncio
import sys
from datetime import datetime
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from app.api.questions import (
    delete_question,
    get_question,
    get_question_image,
    list_questions,
    update_question,
)
from app.schemas.question import QuestionUpdate
from app.schemas.sheet import SheetCreate


def _load_create_sheet():
    previous_pdf = sys.modules.get("app.services.pdf")
    previous_sheets = sys.modules.get("app.api.sheets")
    api_package = sys.modules["app.api"]
    previous_sheets_attribute = getattr(api_package, "sheets", None)

    pdf_module = ModuleType("app.services.pdf")
    pdf_module.generate_sheet_pdf = lambda **_kwargs: ""
    sys.modules["app.services.pdf"] = pdf_module
    sys.modules.pop("app.api.sheets", None)
    try:
        from app.api.sheets import create_sheet
    finally:
        if previous_pdf is None:
            sys.modules.pop("app.services.pdf", None)
        else:
            sys.modules["app.services.pdf"] = previous_pdf
        sys.modules.pop("app.api.sheets", None)
        if previous_sheets is not None:
            sys.modules["app.api.sheets"] = previous_sheets
        if previous_sheets_attribute is None:
            api_package.__dict__.pop("sheets", None)
        else:
            api_package.sheets = previous_sheets_attribute
    return create_sheet


create_sheet = _load_create_sheet()


class EmptyRows:
    def one_or_none(self):
        return None

    def scalar_one_or_none(self):
        return None

    def scalars(self):
        return self

    def all(self):
        return []


class CapturingDB:
    def __init__(self):
        self.queries = []

    async def execute(self, query):
        self.queries.append(query)
        return EmptyRows()


def _compiled_sql(query):
    return str(query.compile(dialect=postgresql.dialect()))


def test_list_questions_excludes_soft_deleted_records():
    db = CapturingDB()

    asyncio.run(
        list_questions(
            subject=None,
            grade=None,
            semester=None,
            status=None,
            tag=None,
            limit=20,
            offset=0,
            created_from=None,
            student_id="student-id",
            db=db,
        )
    )

    assert "wrong_questions.deleted_at IS NULL" in _compiled_sql(db.queries[0])


@pytest.mark.parametrize(
    ("endpoint", "args"),
    [
        (get_question, ("question-id",)),
        (get_question_image, ("question-id",)),
        (update_question, ("question-id", QuestionUpdate(difficulty=2))),
    ],
)
def test_question_endpoints_exclude_soft_deleted_records(endpoint, args):
    db = CapturingDB()

    with pytest.raises(HTTPException):
        asyncio.run(endpoint(*args, student_id="student-id", db=db))

    assert "wrong_questions.deleted_at IS NULL" in _compiled_sql(db.queries[0])


def test_create_sheet_rejects_soft_deleted_questions():
    class SheetDB(CapturingDB):
        async def execute(self, query):
            self.queries.append(query)
            if len(self.queries) == 1:
                return SimpleNamespace(scalar_one=lambda: SimpleNamespace(nickname="student"))
            return EmptyRows()

    db = SheetDB()

    with pytest.raises(HTTPException, match="Some selected questions are unavailable"):
        asyncio.run(
            create_sheet(
                SheetCreate(question_ids=["question-id"]),
                student_id="student-id",
                db=db,
            )
        )

    assert "wrong_questions.deleted_at IS NULL" in _compiled_sql(db.queries[1])


class DeletionDB:
    def __init__(self, question):
        self.question = question
        self.queries = []
        self.delete_calls = 0
        self.commit_calls = 0

    async def execute(self, query):
        self.queries.append(query)
        return SimpleNamespace(scalar_one_or_none=lambda: self.question)

    async def delete(self, question):
        self.delete_calls += 1

    async def commit(self):
        self.commit_calls += 1


def test_delete_question_marks_record_deleted_instead_of_deleting_row():
    question = SimpleNamespace(deleted_at=None)
    db = DeletionDB(question)
    question_id = str(uuid4())
    student_id = str(uuid4())

    asyncio.run(delete_question(question_id, student_id=student_id, db=db))

    assert isinstance(question.deleted_at, datetime)
    assert question.deleted_at.tzinfo is None
    assert db.delete_calls == 0
    assert db.commit_calls == 1
    compiled = db.queries[0].compile(dialect=postgresql.dialect())
    query_sql = str(compiled)
    assert "wrong_questions.id =" in query_sql
    assert "wrong_questions.student_id =" in query_sql
    assert question_id in compiled.params.values()
    assert student_id in compiled.params.values()


def test_delete_question_returns_404_when_no_owned_question_is_found():
    db = DeletionDB(None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(delete_question(str(uuid4()), student_id=str(uuid4()), db=db))

    assert exc_info.value.status_code == 404
    assert db.commit_calls == 0


def test_repeated_delete_keeps_existing_soft_delete_timestamp_and_succeeds():
    deleted_at = datetime(2026, 7, 1, 0, 0, 0)
    question = SimpleNamespace(deleted_at=deleted_at)
    db = DeletionDB(question)

    assert asyncio.run(delete_question("question-id", student_id="student-id", db=db)) == {"ok": True}
    assert question.deleted_at == deleted_at
    assert db.delete_calls == 0
    assert db.commit_calls == 1


def test_confirming_reviewed_question_ignores_soft_deleted_siblings():
    question = SimpleNamespace(status="needs_review", image_id=str(uuid4()), difficulty=None)
    image = SimpleNamespace(status="needs_review")

    class ReviewUpdateDB:
        def __init__(self):
            self.scalar_queries = []

        async def execute(self, _query):
            return SimpleNamespace(scalar_one_or_none=lambda: question)

        async def scalar(self, query):
            self.scalar_queries.append(query)
            if len(self.scalar_queries) == 1:
                return image
            return None

        async def flush(self):
            pass

        async def commit(self):
            pass

    db = ReviewUpdateDB()

    asyncio.run(
        update_question(
            str(uuid4()),
            QuestionUpdate(difficulty=2),
            student_id=str(uuid4()),
            db=db,
        )
    )

    assert "wrong_questions.deleted_at IS NULL" in _compiled_sql(db.scalar_queries[1])
