import asyncio
from types import SimpleNamespace

from app.api.questions import list_questions


class _EmptyResult:
    def scalars(self):
        return self

    def all(self):
        return []


class _CapturingDb:
    def __init__(self):
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return _EmptyResult()


def test_question_pagination_uses_id_to_break_created_at_ties():
    db = _CapturingDb()

    asyncio.run(
        list_questions(
            subject=None,
            grade=None,
            semester=None,
            status=None,
            mastery_status=None,
            tag=None,
            limit=20,
            offset=20,
            created_from=None,
            student=SimpleNamespace(id="student-id"),
            db=db,
        )
    )

    order_by = [str(clause) for clause in db.statement._order_by_clauses]
    assert order_by == [
        "wrong_questions.created_at DESC",
        "wrong_questions.id DESC",
    ]
