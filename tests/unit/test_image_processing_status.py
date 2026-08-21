import asyncio
import inspect
import uuid
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

    assert (code, message) == ("recognition_internal_error", "识别暂时失败，请稍后重试")


def test_processing_failure_preserves_a_safe_recognition_error_category():
    from app.services.vision_recognition import VisionRecognitionError
    from app.tasks.process_image import processing_failure_for

    error = VisionRecognitionError(
        "vision_timeout",
        "识别服务响应超时，请稍后重试",
    )

    assert processing_failure_for(error) == (
        "vision_timeout",
        "识别服务响应超时，请稍后重试",
    )


class RetryDB:
    def __init__(self, image):
        self.image = image
        self.committed = False

    async def scalar(self, _query):
        return self.image

    async def commit(self):
        self.committed = True


class IncompleteImagesDB:
    def __init__(self, images):
        self.images = images
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self.images))


def test_incomplete_image_statuses_exclude_confirmed_and_other_students():
    from app.api import upload as upload_api

    student_id = uuid.uuid4()
    db = IncompleteImagesDB([
        SimpleNamespace(
            id="failed-image",
            status="failed",
            question_count=0,
            error_code="vision_timeout",
            error_message="识别服务响应超时，请稍后重试",
        ),
    ])

    result = asyncio.run(
        upload_api.get_incomplete_image_statuses(
            student=SimpleNamespace(id=student_id),
            db=db,
        )
    )

    assert result == [{
        "image_id": "failed-image",
        "status": "failed",
        "question_count": 0,
        "error_code": "vision_timeout",
        "error_message": "识别服务响应超时，请稍后重试",
    }]
    compiled = str(db.statement.compile(compile_kwargs={"literal_binds": True}))
    assert student_id.hex in compiled
    for status in ("pending", "segmented", "needs_review", "failed"):
        assert status in compiled
    assert "confirmed" not in compiled


def test_incomplete_image_statuses_limit_the_initial_restore_payload():
    from app.api import upload as upload_api

    source = inspect.getsource(upload_api.get_incomplete_image_statuses)

    assert "limit(" in source


def test_cancelling_review_image_ignores_pending_candidates_and_keeps_collected():
    from app.api.upload import cancel_image_tasks

    image = SimpleNamespace(
        id="image-1",
        status="needs_review",
        error_code="vision_timeout",
        error_message="识别服务响应超时，请稍后重试",
    )
    pending_question = SimpleNamespace(
        image_id="image-1",
        collection_status="pending_review",
    )
    collected_question = SimpleNamespace(
        image_id="image-1",
        collection_status="collected",
    )

    cancelled = cancel_image_tasks(
        [image],
        [pending_question, collected_question],
    )

    assert cancelled == ["image-1"]
    assert image.status == "cancelled"
    assert image.error_code is None
    assert image.error_message is None
    assert pending_question.collection_status == "ignored"
    assert collected_question.collection_status == "collected"


def test_cancelling_skips_images_that_are_already_terminal():
    from app.api.upload import cancel_image_tasks

    image = SimpleNamespace(
        id="image-1",
        status="confirmed",
        error_code=None,
        error_message=None,
    )

    assert cancel_image_tasks([image], []) == []
    assert image.status == "confirmed"


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
