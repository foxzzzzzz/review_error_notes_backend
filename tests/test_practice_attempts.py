"""PostgreSQL integration tests for Phase 3 practice-result consistency."""

import json
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.config import settings


def _sync_database_url():
    return make_url(settings.DATABASE_URL).set(
        drivername="postgresql+psycopg2"
    )


@pytest.fixture
def practice_rows(test_user):
    image_id = uuid4()
    question_id = uuid4()
    sheet_ids = [uuid4(), uuid4()]
    item_ids = [uuid4(), uuid4()]
    snapshot = json.dumps(
        {
            "question_type": "original",
            "question_text": "1 + 1 = ?",
            "source_wrong_question_id": str(question_id),
            "sort_order": 0,
        }
    )
    engine = create_engine(_sync_database_url())
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO wrong_images (
                    id, student_id, original_url, grade, semester,
                    question_count, status
                ) VALUES (
                    :id, :student_id, :original_url, 1, 1, 1, 'confirmed'
                )
                """
            ),
            {
                "id": image_id,
                "student_id": test_user["student_id"],
                "original_url": f"/uploads/{image_id}.jpg",
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO wrong_questions (
                    id, student_id, image_id, grade, semester, wrong_count,
                    review_status, mastery_status
                ) VALUES (
                    :id, :student_id, :image_id, 1, 1, 1,
                    'confirmed', 'learning'
                )
                """
            ),
            {
                "id": question_id,
                "student_id": test_user["student_id"],
                "image_id": image_id,
            },
        )
        for sheet_id, item_id in zip(sheet_ids, item_ids):
            connection.execute(
                text(
                    """
                    INSERT INTO practice_sheets (id, student_id, title)
                    VALUES (:id, :student_id, 'Phase 3 integration')
                    """
                ),
                {
                    "id": sheet_id,
                    "student_id": test_user["student_id"],
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO sheet_items (
                        id, sheet_id, wrong_question_id, question_type,
                        question_text, question_snapshot, sort_order
                    ) VALUES (
                        :id, :sheet_id, :question_id, 'original',
                        '1 + 1 = ?', CAST(:snapshot AS jsonb), 0
                    )
                    """
                ),
                {
                    "id": item_id,
                    "sheet_id": sheet_id,
                    "question_id": question_id,
                    "snapshot": snapshot,
                },
            )
    try:
        yield {
            "question_id": question_id,
            "sheet_ids": sheet_ids,
            "item_ids": item_ids,
            "engine": engine,
        }
    finally:
        engine.dispose()


def _create_attempt(client, headers, sheet_id, item_id, is_correct, key):
    return client.post(
        f"/api/sheets/{sheet_id}/attempts",
        headers=headers,
        json={
            "idempotency_key": key,
            "completed_at": "2026-07-25T18:00:00+08:00",
            "items": [
                {
                    "sheet_item_id": str(item_id),
                    "is_correct": is_correct,
                }
            ],
        },
    )


def _question_state(rows):
    with rows["engine"].connect() as connection:
        return connection.execute(
            text(
                """
                SELECT mastery_status::text, wrong_count
                FROM wrong_questions
                WHERE id = :question_id
                """
            ),
            {"question_id": rows["question_id"]},
        ).one()


def test_older_sheet_edit_does_not_override_global_latest_mastery(
    client,
    auth_header,
    practice_rows,
):
    first = _create_attempt(
        client,
        auth_header,
        practice_rows["sheet_ids"][0],
        practice_rows["item_ids"][0],
        True,
        f"first-{uuid4()}",
    )
    assert first.status_code == 200, first.text
    second = _create_attempt(
        client,
        auth_header,
        practice_rows["sheet_ids"][1],
        practice_rows["item_ids"][1],
        True,
        f"second-{uuid4()}",
    )
    assert second.status_code == 200, second.text

    first_attempt = first.json()
    updated = client.patch(
        (
            f"/api/sheets/{practice_rows['sheet_ids'][0]}/attempts/"
            f"{first_attempt['id']}"
        ),
        headers=auth_header,
        json={
            "updated_at": first_attempt["updated_at"],
            "items": [
                {
                    "sheet_item_id": str(practice_rows["item_ids"][0]),
                    "is_correct": False,
                }
            ],
        },
    )

    assert updated.status_code == 200, updated.text
    mastery_status, wrong_count = _question_state(practice_rows)
    assert mastery_status == "mastered"
    assert wrong_count == 2


def test_concurrent_create_and_patch_leave_latest_attempt_mastered(
    client,
    auth_header,
    practice_rows,
):
    sheet_id = practice_rows["sheet_ids"][0]
    item_id = practice_rows["item_ids"][0]
    first = _create_attempt(
        client,
        auth_header,
        sheet_id,
        item_id,
        True,
        f"initial-{uuid4()}",
    )
    assert first.status_code == 200, first.text
    first_attempt = first.json()

    def patch_first():
        return client.patch(
            f"/api/sheets/{sheet_id}/attempts/{first_attempt['id']}",
            headers=auth_header,
            json={
                "updated_at": first_attempt["updated_at"],
                "items": [
                    {
                        "sheet_item_id": str(item_id),
                        "is_correct": False,
                    }
                ],
            },
        )

    def create_second():
        return _create_attempt(
            client,
            auth_header,
            sheet_id,
            item_id,
            True,
            f"concurrent-{uuid4()}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        patch_future = executor.submit(patch_first)
        create_future = executor.submit(create_second)
        patch_response = patch_future.result()
        create_response = create_future.result()

    assert patch_response.status_code in {200, 409}, patch_response.text
    assert create_response.status_code == 200, create_response.text
    mastery_status, _wrong_count = _question_state(practice_rows)
    assert mastery_status == "mastered"


def test_attempt_history_supports_pagination(
    client,
    auth_header,
    practice_rows,
):
    sheet_id = practice_rows["sheet_ids"][0]
    item_id = practice_rows["item_ids"][0]
    for index in range(3):
        response = _create_attempt(
            client,
            auth_header,
            sheet_id,
            item_id,
            index % 2 == 0,
            f"history-{index}-{uuid4()}",
        )
        assert response.status_code == 200, response.text

    response = client.get(
        f"/api/sheets/{sheet_id}/attempts?limit=1&offset=1",
        headers=auth_header,
    )

    assert response.status_code == 200, response.text
    assert len(response.json()) == 1
    assert response.json()[0]["attempt_no"] == 2
    assert len(response.json()[0]["items"]) == 1
