from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_question_response_does_not_expose_source_image_url():
    source = (ROOT / "app" / "schemas" / "question.py").read_text(encoding="utf-8")

    assert "crop_region: Optional[dict] = None" in source
    assert "image_url: Optional[str] = None" not in source


def test_question_detail_joins_the_owning_image():
    source = (ROOT / "app" / "api" / "questions.py").read_text(encoding="utf-8")

    assert "from app.models.wrong_image import WrongImage" in source
    assert ".join(WrongImage" in source
    assert '"image_url": image.original_url' not in source
