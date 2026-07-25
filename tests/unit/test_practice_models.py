from app.models.practice_attempt import PracticeAttempt
from app.models.practice_result import PracticeResult
from app.models.sheet_item import SheetItem


def _unique_column_sets(table):
    return {
        tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }


def test_practice_attempt_has_student_scoped_idempotency_key():
    constraints = _unique_column_sets(PracticeAttempt.__table__)
    assert ("student_id", "idempotency_key") in constraints
    assert ("sheet_id", "attempt_no") in constraints


def test_practice_result_is_unique_per_attempt_item():
    assert ("attempt_id", "sheet_item_id") in _unique_column_sets(
        PracticeResult.__table__
    )
    assert PracticeResult.__table__.c.wrong_question_id.index is True


def test_sheet_item_stores_immutable_question_snapshot():
    assert SheetItem.__table__.c.question_snapshot.nullable is False
