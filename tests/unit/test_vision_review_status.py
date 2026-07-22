from pathlib import Path


BACKEND_ROOT = Path(__file__).parents[2]


def test_models_and_migration_declare_needs_review_status():
    question_model = (BACKEND_ROOT / "app" / "models" / "wrong_question.py").read_text(encoding="utf-8")
    image_model = (BACKEND_ROOT / "app" / "models" / "wrong_image.py").read_text(encoding="utf-8")
    migration = BACKEND_ROOT / "alembic" / "versions" / "0002_add_vision_review_status.py"

    assert '"needs_review"' in question_model
    assert '"needs_review"' in image_model
    assert migration.exists()
    source = migration.read_text(encoding="utf-8")
    assert "ALTER TYPE question_status_enum ADD VALUE IF NOT EXISTS 'needs_review'" in source
    assert "ALTER TYPE image_status_enum ADD VALUE IF NOT EXISTS 'needs_review'" in source


def test_question_update_confirms_reviewed_item_and_completed_image():
    source = (BACKEND_ROOT / "app" / "api" / "questions.py").read_text(encoding="utf-8")

    assert 'was_needs_review = q.status == "needs_review"' in source
    assert 'q.status = "confirmed"' in source
    assert 'WrongQuestion.status == "needs_review"' in source
    assert 'image.status = "confirmed"' in source
    assert "select(WrongImage)" in source
    assert ".with_for_update()" in source
