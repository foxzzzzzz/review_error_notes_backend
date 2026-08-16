import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def test_status_payload_uses_safe_failure_message():
    from app.api.upload import serialize_image_status

    payload = serialize_image_status(
        SimpleNamespace(
            id="image-1",
            status="failed",
            question_count=0,
            error_code="recognition_failed",
            error_message="图片识别失败，请重试",
        )
    )

    assert payload == {
        "image_id": "image-1",
        "status": "failed",
        "question_count": 0,
        "error_code": "recognition_failed",
        "error_message": "图片识别失败，请重试",
    }


def test_processing_failure_hides_internal_exception_details():
    from app.tasks.process_image import processing_failure_for

    code, message = processing_failure_for(RuntimeError("secret-token /uploads/a.jpg"))

    assert (code, message) == ("recognition_failed", "图片识别失败，请重试")


class RetryDB:
    def __init__(self, image):
        self.image = image
        self.committed = False

    async def scalar(self, _query):
        return self.image

    async def commit(self):
        self.committed = True


def test_retry_only_requeues_a_failed_image(monkeypatch):
    from app.api import upload as upload_api

    image = SimpleNamespace(
        id="image-1",
        student_id="student-1",
        status="failed",
        original_url="/uploads/source.jpg",
        error_code="recognition_failed",
        error_message="图片识别失败，请重试",
    )
    delayed = []
    monkeypatch.setattr(upload_api.settings, "UPLOAD_DIR", "/uploads")
    monkeypatch.setattr(upload_api.process_image, "delay", lambda *args: delayed.append(args))

    response = asyncio.run(
        upload_api.retry_image(
            image_id="image-1",
            student=SimpleNamespace(id="student-1"),
            db=RetryDB(image),
        )
    )

    assert response == {"image_id": "image-1", "status": "pending"}
    assert image.status == "pending"
    assert image.error_code is None
    assert image.error_message is None
    assert delayed == [("image-1", str(Path("/uploads") / "source.jpg"))]


def test_retry_rejects_an_image_that_is_not_failed():
    from app.api import upload as upload_api

    image = SimpleNamespace(
        id="image-1",
        student_id="student-1",
        status="confirmed",
        original_url="/uploads/source.jpg",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            upload_api.retry_image(
                image_id="image-1",
                student=SimpleNamespace(id="student-1"),
                db=RetryDB(image),
            )
        )

    assert exc_info.value.status_code == 409
