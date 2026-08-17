from types import SimpleNamespace
from pathlib import Path


SHEETS_API = Path(__file__).parents[2] / "app" / "api" / "sheets.py"


def test_order_questions_follows_submitted_question_ids():
    from app.services.practice_question import order_questions

    first = SimpleNamespace(id="first")
    second = SimpleNamespace(id="second")

    assert order_questions([first, second], ["second", "first"]) == [second, first]


def test_order_questions_ignores_unmatched_records():
    from app.services.practice_question import order_questions

    first = SimpleNamespace(id="first")

    assert order_questions([first], ["missing", "first"]) == [first]


def test_sheet_api_exposes_transactional_practice_result_routes():
    source = SHEETS_API.read_text(encoding="utf-8")

    assert '@router.get("/{sheet_id}/review"' in source
    assert '@router.get("/{sheet_id}/attempts"' in source
    assert '@router.post("/{sheet_id}/attempts"' in source
    assert '"/{sheet_id}/attempts/{attempt_id}"' in source
    assert "with_for_update()" in source
    assert "PracticeResultValidationError" in source
    assert "attempt_conflict" in source
