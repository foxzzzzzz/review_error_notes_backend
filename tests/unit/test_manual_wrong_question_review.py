import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.questions import add_manual_review_question
from app.schemas.question import ManualWrongQuestionRequest
from app.services.chinese_marked_evidence import PIPELINE_NAME


class _Db:
    def __init__(self, image, pending=None):
        self.image = image
        self.pending = pending
        self.questions = {}
        self.added = []
        self.commit_calls = 0

    async def scalar(self, statement):
        if "wrong_images" in str(statement):
            return self.image
        return self.pending

    async def get(self, _model, question_id):
        return self.questions.get(question_id)

    def add(self, question):
        self.added.append(question)
        self.questions[question.id] = question

    async def commit(self):
        self.commit_calls += 1


def _request(question_id=None, **overrides):
    values = {
        "question_id": question_id or uuid4(),
        "bbox": [0.1, 0.2, 0.7, 0.8],
        "instruction": "看拼音写词语",
        "prompt_text": "yǎn jìng",
        "question_type": "write_word",
        "correct_answer": "眼镜",
        "student_answer": "眼睛",
    }
    values.update(overrides)
    return ManualWrongQuestionRequest(**values)


def _image(**overrides):
    values = {
        "id": uuid4(),
        "student_id": uuid4(),
        "status": "confirmed",
        "question_count": 2,
        "error_code": None,
        "error_message": None,
        "subject": "chinese",
        "grade": 2,
        "semester": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_manual_question_is_collected_with_human_evidence_and_preserves_pending_image():
    image = _image(status="needs_review")
    student = SimpleNamespace(id=image.student_id)
    db = _Db(image, pending="sibling-pending")
    request = _request()

    result = asyncio.run(
        add_manual_review_question(str(image.id), request, student=student, db=db)
    )

    question = db.added[0]
    evidence = question.ocr_raw_json["evidence_bundle"]
    assert result == {"question_id": str(request.question_id), "idempotent": False}
    assert question.collection_status == "collected"
    assert question.ocr_text == "眼睛"
    assert question.crop_region == {"bbox_format": "normalized_ltrb", "bbox": request.bbox}
    assert question.recognition_pipeline == PIPELINE_NAME
    assert evidence["human_confirmed_prompt"]["source"] == "human"
    assert evidence["human_confirmed_prompt"]["actor_id"] == str(student.id)
    assert evidence["review_history"][0]["decision"] == "manual_add"
    assert image.question_count == 3
    assert image.status == "needs_review"


def test_manual_add_finishes_image_issue_and_keeps_image_in_completed_history():
    image = _image(status="needs_review", question_count=0, error_code="no_candidates", error_message="没有候选题")
    result = asyncio.run(
        add_manual_review_question(
            str(image.id),
            _request(),
            student=SimpleNamespace(id=image.student_id),
            db=_Db(image),
        )
    )

    assert result["idempotent"] is False
    assert image.status == "confirmed"
    assert image.question_count == 1
    assert image.error_code is None
    assert image.error_message is None


def test_manual_question_retry_is_idempotent_for_same_payload_and_conflicts_for_changes():
    image = _image()
    student = SimpleNamespace(id=image.student_id)
    db = _Db(image)
    request = _request()
    first = asyncio.run(
        add_manual_review_question(str(image.id), request, student=student, db=db)
    )
    second = asyncio.run(
        add_manual_review_question(str(image.id), request, student=student, db=db)
    )

    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert len(db.added) == 1
    assert db.commit_calls == 1

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            add_manual_review_question(
                str(image.id),
                _request(question_id=request.question_id, correct_answer="眼睛"),
                student=student,
                db=db,
            )
        )
    assert exc_info.value.status_code == 409


@pytest.mark.parametrize(
    "overrides",
    [
        {"bbox": [0.8, 0.2, 0.7, 0.8]},
        {"bbox": [0, 0, 1, float("nan")]},
        {"question_type": "unknown"},
        {"correct_answer": "   "},
    ],
)
def test_manual_question_rejects_invalid_bbox_or_fields(overrides):
    image = _image()
    student = SimpleNamespace(id=image.student_id)
    db = _Db(image)
    request = _request(**overrides)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            add_manual_review_question(str(image.id), request, student=student, db=db)
        )
    assert exc_info.value.status_code == 422
    assert not db.added


def test_manual_question_rejects_images_not_in_reviewable_states():
    image = _image(status="pending")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            add_manual_review_question(
                str(image.id),
                _request(),
                student=SimpleNamespace(id=image.student_id),
                db=_Db(image),
            )
        )
    assert exc_info.value.status_code == 409
