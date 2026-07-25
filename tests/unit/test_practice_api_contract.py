from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.sheet import AttemptCreate, AttemptUpdate


SHEETS_SOURCE = (
    Path(__file__).parents[2] / "app" / "api" / "sheets.py"
).read_text(encoding="utf-8")


def test_attempt_payload_rejects_duplicate_sheet_item_ids():
    item_id = uuid4()

    with pytest.raises(ValidationError):
        AttemptCreate(
            idempotency_key="one",
            completed_at=datetime(2026, 7, 25, 12, 0),
            items=[
                {"sheet_item_id": item_id, "is_correct": True},
                {"sheet_item_id": item_id, "is_correct": False},
            ],
        )


def test_attempt_update_requires_optimistic_version():
    with pytest.raises(ValidationError):
        AttemptUpdate(items=[])


def test_review_defaults_every_item_to_correct_without_attempt():
    from app.services.practice_results import build_review_groups

    question_id = uuid4()
    items = [
        SimpleNamespace(
            id=uuid4(),
            wrong_question_id=question_id,
            question_type="original",
            question_text="题目",
            question_snapshot={"question_text": "题目"},
            sort_order=0,
        )
    ]

    groups = build_review_groups(items, {})

    assert groups[0]["wrong_question_id"] == question_id
    assert groups[0]["items"][0]["is_correct"] is True


def test_mastery_update_clamps_wrong_count_and_sets_timestamps():
    from app.services.practice_results import apply_group_result

    now = datetime(2026, 7, 25, 12, 0)
    question = SimpleNamespace(
        mastery_status="learning",
        mastered_at=None,
        last_practiced_at=None,
        wrong_count=1,
    )

    apply_group_result(question, all_correct=False, previous_correct=None, now=now)
    assert question.wrong_count == 2
    assert question.mastery_status == "learning"

    apply_group_result(question, all_correct=True, previous_correct=False, now=now)
    assert question.wrong_count == 1
    assert question.mastery_status == "mastered"
    assert question.mastered_at == now
    assert question.last_practiced_at == now


def test_editing_non_latest_attempt_only_adjusts_wrong_count():
    from app.services.practice_results import apply_group_result

    mastered_at = datetime(2026, 7, 25, 13, 0)
    last_practiced_at = datetime(2026, 7, 25, 13, 0)
    question = SimpleNamespace(
        mastery_status="mastered",
        mastered_at=mastered_at,
        last_practiced_at=last_practiced_at,
        wrong_count=2,
    )

    apply_group_result(
        question,
        all_correct=False,
        previous_correct=True,
        now=datetime(2026, 7, 25, 12, 0),
        update_mastery=False,
    )

    assert question.wrong_count == 3
    assert question.mastery_status == "mastered"
    assert question.mastered_at == mastered_at
    assert question.last_practiced_at == last_practiced_at


def test_attempt_routes_share_sheet_lock_and_global_latest_lookup():
    create_source, update_source = SHEETS_SOURCE.split(
        "async def update_sheet_attempt",
        maxsplit=1,
    )

    assert "_owned_sheet(db, student.id, sheet_id, lock=True)" in create_source
    assert "_owned_sheet(db, student.id, sheet_id, lock=True)" in update_source
    assert "_latest_attempt_ids_by_question" in create_source
    assert "_latest_attempt_ids_by_question" in update_source
    assert create_source.count(".order_by(WrongQuestion.id)") >= 1
    assert update_source.count(".order_by(WrongQuestion.id)") >= 1


def test_attempt_history_is_paginated_and_batch_loaded():
    assert "limit: int = Query(20, ge=1, le=100)" in SHEETS_SOURCE
    assert "offset: int = Query(0, ge=0)" in SHEETS_SOURCE
    assert "_attempts_out" in SHEETS_SOURCE
