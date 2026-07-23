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


def test_sheet_api_uses_clean_builder_grouped_pdf_and_explicit_errors():
    source = SHEETS_API.read_text(encoding="utf-8")

    assert "questions = order_questions(" in source
    assert "build_printable_questions(" in source
    assert "status_code=422" in source
    assert "status_code=503" in source
    assert "len(set(data.question_ids)) != len(data.question_ids)" in source
    assert "groups=groups" in source
    assert "remove_generated_pdf(" in source
    assert "original_items=" not in source
    assert "derived_items=" not in source


def test_sheet_api_renders_before_committing_database_rows():
    source = SHEETS_API.read_text(encoding="utf-8")

    render_position = source.index("generate_sheet_pdf(")
    commit_position = source.index("await db.commit()")

    assert render_position < commit_position
