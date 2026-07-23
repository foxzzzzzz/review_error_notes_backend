import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
SOURCE_PATH = ROOT / "app" / "api" / "questions.py"


def _source():
    return SOURCE_PATH.read_text(encoding="utf-8")


def test_question_image_route_has_valid_python_syntax():
    ast.parse(_source())


def test_question_image_route_declares_crop_and_original_views():
    source = _source()

    assert '@router.get("/{question_id}/image")' in source
    assert 'view: Literal["crop", "original"] = "crop"' in source


def test_question_image_route_filters_by_question_and_current_student():
    source = _source()

    assert "WrongQuestion.id == question_id" in source
    assert "WrongQuestion.student_id == student_id" in source
    assert ".join(WrongImage, WrongImage.id == WrongQuestion.image_id)" in source


def test_question_image_route_limits_source_path_to_upload_directory():
    source = _source()

    assert "Path(settings.UPLOAD_DIR) / Path(image.original_url).name" in source


def test_question_image_route_maps_service_errors_and_returns_jpeg():
    source = _source()

    assert "except QuestionImageNotFound:" in source
    assert 'status_code=404, detail="Question image not found"' in source
    assert "except QuestionImageInvalid:" in source
    assert 'status_code=422, detail="Question image is invalid"' in source
    assert "settings.QUESTION_IMAGE_MAX_PIXELS" in source
    assert 'Response(content=content, media_type="image/jpeg")' in source
