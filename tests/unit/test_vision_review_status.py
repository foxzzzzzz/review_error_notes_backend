from pathlib import Path


BACKEND_ROOT = Path(__file__).parents[2]


def test_models_and_initial_schema_declare_separate_review_and_mastery_statuses():
    question_model = (BACKEND_ROOT / "app" / "models" / "wrong_question.py").read_text(encoding="utf-8")
    image_model = (BACKEND_ROOT / "app" / "models" / "wrong_image.py").read_text(encoding="utf-8")
    migration = BACKEND_ROOT / "alembic" / "versions" / "0001_initial_schema.py"

    assert "review_status = Column(" in question_model
    assert "mastery_status = Column(" in question_model
    assert '"needs_review"' in image_model
    source = migration.read_text(encoding="utf-8")
    assert "question_review_status_enum = postgresql.ENUM(" in source
    assert '"needs_review",' in source
    assert "question_mastery_status_enum = postgresql.ENUM(" in source
    assert '"learning",' in source
    assert '"mastered",' in source


def test_question_update_confirms_reviewed_item_and_completed_image():
    source = (BACKEND_ROOT / "app" / "api" / "questions.py").read_text(encoding="utf-8")

    assert 'was_needs_review = q.review_status == "needs_review"' in source
    assert 'q.review_status = "confirmed"' in source
    assert 'WrongQuestion.review_status == "needs_review"' in source
    assert 'image.status = "confirmed"' in source
    assert "select(WrongImage)" in source
    assert ".with_for_update()" in source
