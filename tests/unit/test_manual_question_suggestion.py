from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import questions as question_api


def _test_client(monkeypatch, *, image, student_id=None):
    owner_id = student_id or (image.student_id if image is not None else uuid4())
    student = SimpleNamespace(id=owner_id)

    class FakeDB:
        async def scalar(self, statement):
            sql = str(statement)
            assert "wrong_images.student_id" in sql
            if image is not None and image.student_id == owner_id:
                return image
            return None

    async def get_student():
        return student

    async def get_db():
        yield FakeDB()

    app = FastAPI()
    app.include_router(question_api.router)
    app.dependency_overrides[question_api.get_default_student] = get_student
    app.dependency_overrides[question_api.get_db] = get_db
    return TestClient(app), owner_id


def _image(owner_id, status="needs_review"):
    return SimpleNamespace(
        id=uuid4(),
        student_id=owner_id,
        original_url="/uploads/manual-suggestion.jpg",
        status=status,
        subject="chinese",
        grade=1,
        semester=1,
    )


def test_manual_suggestion_rejects_image_owned_by_another_student(monkeypatch):
    owner_id = uuid4()
    image = _image(owner_id)
    client, _ = _test_client(monkeypatch, image=image, student_id=uuid4())

    response = client.post(
        f"/questions/review/images/{image.id}/manual-suggestion",
        json={"bbox": [0.1, 0.1, 0.9, 0.9], "mode": "ocr"},
    )

    assert response.status_code == 404


def test_manual_suggestion_rejects_invalid_bbox_before_running_ocr(monkeypatch):
    owner_id = uuid4()
    image = _image(owner_id)
    client, _ = _test_client(monkeypatch, image=image)
    monkeypatch.setattr(
        question_api,
        "recognize_manual_suggestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("OCR must not run")),
    )

    response = client.post(
        f"/questions/review/images/{image.id}/manual-suggestion",
        json={"bbox": [0.8, 0.1, 0.2, 0.9], "mode": "ocr"},
    )

    assert response.status_code == 422


def test_manual_suggestion_ocr_returns_text_as_prompt_draft_without_persisting(monkeypatch):
    owner_id = uuid4()
    image = _image(owner_id)
    client, _ = _test_client(monkeypatch, image=image)
    called = {}

    def recognize(filepath, bbox, mode, **kwargs):
        called.update(filepath=filepath, bbox=bbox, mode=mode)
        return {
            "mode": "ocr",
            "fields": {
                "instruction": "",
                "prompt_text": "看拼音写词语",
                "question_type": "",
                "correct_answer": "",
                "student_answer": "",
            },
            "ocr_text": "看拼音写词语",
        }

    monkeypatch.setattr(question_api, "recognize_manual_suggestion", recognize)

    response = client.post(
        f"/questions/review/images/{image.id}/manual-suggestion",
        json={"bbox": [0.1, 0.2, 0.8, 0.6], "mode": "ocr"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "mode": "ocr",
        "fields": {
            "instruction": "",
            "prompt_text": "看拼音写词语",
            "question_type": "",
            "correct_answer": "",
            "student_answer": "",
        },
        "ocr_text": "看拼音写词语",
    }
    assert called["bbox"] == [0.1, 0.2, 0.8, 0.6]
    assert called["mode"] == "ocr"
    assert not hasattr(image, "question_count")


def test_manual_suggestion_llm_keeps_missing_correct_answer_empty(monkeypatch):
    owner_id = uuid4()
    image = _image(owner_id)
    client, _ = _test_client(monkeypatch, image=image)
    monkeypatch.setattr(
        question_api,
        "recognize_manual_suggestion",
        lambda *args, **kwargs: {
            "mode": "llm",
            "fields": {
                "instruction": "看拼音写词语",
                "prompt_text": "shuǐ guǒ",
                "question_type": "write_word",
                "correct_answer": "",
                "student_answer": "水果",
            },
            "ocr_text": "shuǐ guǒ 水果",
        },
    )

    response = client.post(
        f"/questions/review/images/{image.id}/manual-suggestion",
        json={"bbox": [0.1, 0.2, 0.8, 0.6], "mode": "llm"},
    )

    assert response.status_code == 200
    assert response.json()["fields"]["correct_answer"] == ""
    assert response.json()["fields"]["question_type"] == "write_word"


def test_manual_suggestion_does_not_run_for_unavailable_image_status(monkeypatch):
    owner_id = uuid4()
    image = _image(owner_id, status="pending")
    client, _ = _test_client(monkeypatch, image=image)
    monkeypatch.setattr(
        question_api,
        "recognize_manual_suggestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    response = client.post(
        f"/questions/review/images/{image.id}/manual-suggestion",
        json={"bbox": [0.1, 0.2, 0.8, 0.6], "mode": "ocr"},
    )

    assert response.status_code == 409


def test_manual_suggestion_failure_leaves_manual_entry_available(monkeypatch):
    owner_id = uuid4()
    image = _image(owner_id)
    client, _ = _test_client(monkeypatch, image=image)
    monkeypatch.setattr(
        question_api,
        "recognize_manual_suggestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    response = client.post(
        f"/questions/review/images/{image.id}/manual-suggestion",
        json={"bbox": [0.1, 0.2, 0.8, 0.6], "mode": "llm"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Recognition suggestion unavailable; continue manual entry"
    assert not hasattr(image, "question_count")


def test_api_compose_has_manual_suggestion_runtime_dependencies():
    compose = (Path(__file__).parents[2] / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    api_service = compose.split("\n  worker:", 1)[0]

    assert "DEEPSEEK_VISION_API_BASE:" in api_service
    assert "DEEPSEEK_VISION_MODEL:" in api_service
    assert "LOCAL_OCR_MODEL_VERSION:" in api_service
    assert "MANUAL_QUESTION_SUGGESTION_PROMPT_PATH:" in api_service
    assert "./config:/app/config:ro" in api_service
    assert "LOCAL_OCR_MODEL_PATH: ${LOCAL_OCR_MODEL_PATH:-/opt/rapidocr-models}" in api_service
