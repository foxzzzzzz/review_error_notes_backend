import asyncio
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from app.api.questions import (
    _current_evidence_prompt_values,
    decide_image_reviews,
    update_question,
)
from app.schemas.question import QuestionUpdate, ReviewDecision, ReviewDecisionRequest
from app.services.chinese_marked_evidence import PIPELINE_NAME


STUDENT_ID = uuid4()


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self

    def all(self):
        return self.rows

    def scalar_one_or_none(self):
        return self.rows[0] if self.rows else None


class _ReviewDb:
    def __init__(self, image, question, remaining=False):
        self.image = image
        self.question = question
        self.remaining = remaining
        self.non_pending_reads = 0
        self.commit_calls = 0
        self.flush_calls = 0

    async def scalar(self, statement):
        if "wrong_images" in str(statement):
            return self.image
        return "pending-sibling" if self.remaining else None

    async def execute(self, _statement):
        if self.question.collection_status == "pending_review":
            return _Rows([self.question])
        statement_sql = str(_statement.compile(dialect=postgresql.dialect()))
        if (
            self.question.deleted_at is not None
            and "wrong_questions.deleted_at IS NULL"
            in statement_sql
        ):
            return _Rows([])
        if "wrong_questions.collection_status =" in statement_sql:
            return _Rows([])
        return _Rows([self.question])

    async def flush(self):
        self.flush_calls += 1

    async def commit(self):
        self.commit_calls += 1


class _BatchReviewDb:
    def __init__(self, image, questions):
        self.image = image
        self.questions = questions
        self.execute_calls = 0
        self.commit_calls = 0
        self.flush_calls = 0

    async def scalar(self, statement):
        if "wrong_images" in str(statement):
            return self.image
        return None

    async def execute(self, _statement):
        self.execute_calls += 1
        statement_sql = str(_statement.compile(dialect=postgresql.dialect()))
        if "wrong_questions.collection_status =" in statement_sql:
            return _Rows(
                [
                    question
                    for question in self.questions
                    if question.collection_status == "pending_review"
                    and question.deleted_at is None
                ]
            )
        return _Rows([question for question in self.questions if question.deleted_at is None])

    async def flush(self):
        self.flush_calls += 1

    async def commit(self):
        self.commit_calls += 1


def _question(**overrides):
    values = {
        "id": uuid4(),
        "image_id": uuid4(),
        "recognition_pipeline": PIPELINE_NAME,
        "collection_status": "pending_review",
        "review_status": "needs_review",
        "mark_status": "suggested",
        "question_evidence_status": "suggested",
        "answer_status": "suggested",
        "ocr_answer": "候选答案",
        "question_type": "write_word",
        "subject": "chinese",
        "difficulty": 2,
        "ocr_raw_json": {
            "deepseek_content": {
                "instruction": "看拼音写词语",
                "prompt_text": "yǎn jìng",
                "question_type": "write_word",
            },
            "evidence_bundle": {
                "answer_suggestion": "候选答案",
                "mark": {"shape": "underline"},
            }
        },
        "deleted_at": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _image(question):
    return SimpleNamespace(id=question.image_id, status="needs_review")


def _review(question, decision, db=None):
    db = db or _ReviewDb(_image(question), question)
    response = asyncio.run(
        decide_image_reviews(
            str(question.image_id),
            ReviewDecisionRequest(decisions=[decision]),
            student=SimpleNamespace(id=STUDENT_ID),
            db=db,
        )
    )
    return response, db


def test_primary_page_review_prefills_visible_printed_question_when_ocr_conflicts():
    question = _question(question_type=None)
    raw_json = {
        "marked_page_item": {"printed_question": "qiū liáng"},
    }
    bundle = {
        "fields": {
            "printed_prompt": {
                "status": "conflict",
                "selected_value": None,
            },
        },
    }

    values = _current_evidence_prompt_values(question, raw_json, bundle)

    assert values["prompt_text"] == "qiū liáng"
    assert values["instruction"] == ""
    assert values["question_type"] == ""


def test_collect_confirms_all_evidence_and_appends_normalized_audit_record():
    question = _question()
    decision = ReviewDecision(
        question_id=question.id,
        decision="collect",
        correct_answer="  眼镜  ",
        question_type="词语辨析",
        instruction="选择正确词语",
        prompt_text="给带横线词语选正确答案",
    )

    response, db = _review(question, decision)

    assert response == {"collected": 1, "ignored": 0, "remaining": False}
    assert question.collection_status == "collected"
    assert question.review_status == "confirmed"
    assert question.mark_status == "confirmed"
    assert question.question_evidence_status == "confirmed"
    assert question.answer_status == "confirmed"
    assert question.ocr_answer == "眼镜"
    bundle = question.ocr_raw_json["evidence_bundle"]
    assert bundle["answer_suggestion"] == "候选答案"
    assert bundle["human_override"] == {
        "question_type": "词语辨析",
        "instruction": "选择正确词语",
        "prompt_text": "给带横线词语选正确答案",
    }
    assert bundle["human_confirmed_prompt"] == {
        "instruction": "选择正确词语",
        "prompt_text": "给带横线词语选正确答案",
        "question_type": "词语辨析",
        "source": "human",
        "actor_id": str(STUDENT_ID),
    }
    assert question.question_type == "词语辨析"
    assert bundle["review_history"][-1]["source"] == "human"
    assert bundle["review_history"][-1]["actor_id"] == str(STUDENT_ID)
    assert bundle["review_history"][-1]["old_values"]["question_type"] == "write_word"
    assert bundle["review_history"][-1]["old_values"]["instruction"] == "看拼音写词语"
    assert bundle["review_history"][-1]["old_values"]["prompt_text"] == "yǎn jìng"
    assert bundle["review_history"][-1]["new_values"]["correct_answer"] == "眼镜"
    assert "instruction" not in question.ocr_raw_json
    assert "prompt_text" not in question.ocr_raw_json
    assert db.commit_calls == 1


def test_collect_without_prompt_corrections_confirms_current_display_snapshot():
    question = _question()

    _review(
        question,
        ReviewDecision(
            question_id=question.id,
            decision="collect",
            correct_answer="眼镜",
        ),
    )

    bundle = question.ocr_raw_json["evidence_bundle"]
    assert bundle["human_confirmed_prompt"] == {
        "instruction": "看拼音写词语",
        "prompt_text": "yǎn jìng",
        "question_type": "write_word",
        "source": "human",
        "actor_id": str(STUDENT_ID),
    }
    assert bundle.get("human_override") is None


@pytest.mark.parametrize("field", ["question_type", "instruction", "prompt_text"])
def test_collect_rejects_explicit_blank_prompt_correction(field):
    question = _question()
    kwargs = {
        "question_id": question.id,
        "decision": "collect",
        "correct_answer": "眼镜",
        field: "   ",
    }
    db = _ReviewDb(_image(question), question)

    with pytest.raises(HTTPException) as exc_info:
        _review(question, ReviewDecision(**kwargs), db)

    assert exc_info.value.status_code == 422
    assert db.flush_calls == 0
    assert db.commit_calls == 0


def test_collect_rejects_missing_current_prompt_without_correction():
    question = _question(ocr_raw_json={"evidence_bundle": {}})
    db = _ReviewDb(_image(question), question)

    with pytest.raises(HTTPException) as exc_info:
        _review(
            question,
            ReviewDecision(
                question_id=question.id,
                decision="collect",
                correct_answer="眼镜",
            ),
            db,
        )

    assert exc_info.value.status_code == 422
    assert db.commit_calls == 0


def test_ignore_keeps_suggested_answer_while_confirming_review():
    question = _question()

    _review(question, ReviewDecision(question_id=question.id, decision="ignore"))

    assert question.collection_status == "ignored"
    assert question.review_status == "confirmed"
    assert question.ocr_answer == "候选答案"
    assert question.answer_status == "suggested"


def test_legacy_collect_without_answer_keeps_existing_decision_behavior():
    question = _question(recognition_pipeline="legacy_ocr_v2")

    _review(question, ReviewDecision(question_id=question.id, decision="collect"))

    assert question.collection_status == "collected"
    assert question.review_status == "confirmed"
    assert question.ocr_answer == "候选答案"


@pytest.mark.parametrize("correct_answer", [None, "  "])
def test_new_pipeline_collect_without_nonblank_answer_is_rejected_before_writes(
    correct_answer,
):
    question = _question()
    original_raw_json = deepcopy(question.ocr_raw_json)
    db = _ReviewDb(_image(question), question)

    with pytest.raises(HTTPException) as exc_info:
        _review(
            question,
            ReviewDecision(
                question_id=question.id,
                decision="collect",
                correct_answer=correct_answer,
            ),
            db,
        )

    assert exc_info.value.status_code == 422
    assert question.collection_status == "pending_review"
    assert question.review_status == "needs_review"
    assert question.ocr_raw_json == original_raw_json
    assert db.flush_calls == 0
    assert db.commit_calls == 0


def test_mixed_terminal_retry_and_pending_decision_succeed_in_one_transaction():
    terminal_question = _question()
    _review(
        terminal_question,
        ReviewDecision(
            question_id=terminal_question.id,
            decision="collect",
            correct_answer="眼镜",
        ),
    )
    pending_question = _question(image_id=terminal_question.image_id)
    db = _BatchReviewDb(_image(terminal_question), [terminal_question, pending_question])

    response = asyncio.run(
        decide_image_reviews(
            str(terminal_question.image_id),
            ReviewDecisionRequest(
                decisions=[
                    ReviewDecision(
                        question_id=terminal_question.id,
                        decision="collect",
                        correct_answer="眼镜",
                    ),
                    ReviewDecision(question_id=pending_question.id, decision="ignore"),
                ]
            ),
            student=SimpleNamespace(id=uuid4()),
            db=db,
        )
    )

    assert response == {"collected": 1, "ignored": 1, "remaining": False}
    assert len(terminal_question.ocr_raw_json["evidence_bundle"]["review_history"]) == 1
    assert pending_question.collection_status == "ignored"
    assert len(pending_question.ocr_raw_json["evidence_bundle"]["review_history"]) == 1
    assert db.commit_calls == 1


def test_terminal_conflict_prevents_writes_or_audit_for_pending_questions():
    terminal_question = _question()
    _review(
        terminal_question,
        ReviewDecision(
            question_id=terminal_question.id,
            decision="collect",
            correct_answer="眼镜",
        ),
    )
    pending_question = _question(image_id=terminal_question.image_id)
    terminal_history = deepcopy(
        terminal_question.ocr_raw_json["evidence_bundle"]["review_history"]
    )
    pending_raw_json = deepcopy(pending_question.ocr_raw_json)
    db = _BatchReviewDb(_image(terminal_question), [terminal_question, pending_question])

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            decide_image_reviews(
                str(terminal_question.image_id),
                ReviewDecisionRequest(
                    decisions=[
                        ReviewDecision(
                            question_id=terminal_question.id,
                            decision="collect",
                            correct_answer="雨伞",
                        ),
                        ReviewDecision(
                            question_id=pending_question.id,
                            decision="collect",
                            correct_answer="新答案",
                        ),
                    ]
                ),
                student=SimpleNamespace(id=uuid4()),
                db=db,
            )
        )

    assert exc_info.value.status_code == 409
    assert terminal_question.ocr_raw_json["evidence_bundle"]["review_history"] == terminal_history
    assert pending_question.collection_status == "pending_review"
    assert pending_question.ocr_raw_json == pending_raw_json
    assert db.flush_calls == 0
    assert db.commit_calls == 0


def test_identical_terminal_retry_is_idempotent_without_duplicate_audit():
    question = _question()
    first = ReviewDecision(
        question_id=question.id,
        decision="collect",
        correct_answer="眼镜",
    )
    _, db = _review(question, first)
    initial_history = deepcopy(question.ocr_raw_json["evidence_bundle"]["review_history"])

    response, _ = _review(
        question,
        ReviewDecision(
            question_id=question.id,
            decision="collect",
            correct_answer="  眼镜  ",
        ),
        db,
    )

    assert response == {"collected": 1, "ignored": 0, "remaining": False}
    assert question.ocr_raw_json["evidence_bundle"]["review_history"] == initial_history
    assert db.commit_calls == 1


def test_identical_terminal_retry_preserves_remaining_review_result():
    question = _question()
    db = _ReviewDb(_image(question), question, remaining=True)
    decision = ReviewDecision(
        question_id=question.id,
        decision="collect",
        correct_answer="眼镜",
    )

    first_response, _ = _review(question, decision, db)
    retry_response, _ = _review(question, decision, db)

    assert first_response["remaining"] is True
    assert retry_response == first_response


@pytest.mark.parametrize(
    "decision",
    [
        lambda question: ReviewDecision(
            question_id=question.id, decision="collect", correct_answer="雨伞"
        ),
        lambda question: ReviewDecision(question_id=question.id, decision="ignore"),
    ],
)
def test_terminal_retry_with_changed_decision_or_answer_conflicts(decision):
    question = _question()
    _, db = _review(
        question,
        ReviewDecision(question_id=question.id, decision="collect", correct_answer="眼镜"),
    )

    with pytest.raises(HTTPException) as exc_info:
        _review(question, decision(question), db)

    assert exc_info.value.status_code == 409
    assert db.commit_calls == 1


def test_soft_deleted_terminal_question_is_not_idempotently_replayed():
    question = _question(
        collection_status="collected",
        review_status="confirmed",
        deleted_at=datetime(2026, 9, 15),
    )
    question.ocr_raw_json["evidence_bundle"]["review_history"] = [
        {
            "decision": "collect",
            "new_values": {
                "correct_answer": "眼镜",
                "question_type": None,
                "instruction": None,
                "prompt_text": None,
            },
        }
    ]

    with pytest.raises(HTTPException) as exc_info:
        _review(
            question,
            ReviewDecision(question_id=question.id, decision="collect", correct_answer="眼镜"),
        )

    assert exc_info.value.status_code == 404
    assert question.collection_status == "collected"


def test_active_superseded_question_returns_conflict_without_revival():
    question = _question(collection_status="superseded", review_status="confirmed")

    with pytest.raises(HTTPException) as exc_info:
        _review(
            question,
            ReviewDecision(question_id=question.id, decision="collect", correct_answer="眼镜"),
        )

    assert exc_info.value.status_code == 409
    assert question.collection_status == "superseded"


def test_patch_cannot_confirm_a_new_pipeline_question():
    question = _question()

    class _UpdateDb:
        commit_calls = 0

        async def execute(self, _statement):
            return _Rows([question])

        async def commit(self):
            self.commit_calls += 1

    db = _UpdateDb()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            update_question(
                str(question.id),
                QuestionUpdate(review_status="confirmed"),
                student=SimpleNamespace(id=uuid4()),
                db=db,
            )
        )

    assert exc_info.value.status_code == 409
    assert question.review_status == "needs_review"
    assert db.commit_calls == 0


def test_patch_cannot_overwrite_student_ocr_text_in_new_pipeline():
    question = _question(ocr_text="学生原作答")

    class _UpdateDb:
        commit_calls = 0

        async def execute(self, _statement):
            return _Rows([question])

        async def commit(self):
            self.commit_calls += 1

    db = _UpdateDb()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            update_question(
                str(question.id),
                QuestionUpdate(ocr_text="人工覆盖"),
                student=SimpleNamespace(id=STUDENT_ID),
                db=db,
            )
        )

    assert exc_info.value.status_code == 409
    assert question.ocr_text == "学生原作答"
    assert db.commit_calls == 0


def test_patch_allows_non_review_metadata_without_changing_audit():
    question = _question()
    original_bundle = deepcopy(question.ocr_raw_json["evidence_bundle"])

    class _UpdateDb:
        commit_calls = 0

        async def execute(self, _statement):
            return _Rows([question])

        async def commit(self):
            self.commit_calls += 1

    db = _UpdateDb()

    response = asyncio.run(
        update_question(
            str(question.id),
            QuestionUpdate(subject="语文", difficulty=3),
            student=SimpleNamespace(id=STUDENT_ID),
            db=db,
        )
    )

    assert response == {"ok": True}
    assert question.subject == "语文"
    assert question.difficulty == 3
    assert question.ocr_raw_json["evidence_bundle"] == original_bundle
    assert db.commit_calls == 1


def test_legacy_review_keeps_existing_decision_behavior():
    question = _question(recognition_pipeline="legacy_ocr_v2")

    _review(
        question,
        ReviewDecision(question_id=question.id, decision="collect", correct_answer="眼镜"),
    )

    assert question.collection_status == "collected"
    assert question.review_status == "confirmed"
    assert question.ocr_answer == "候选答案"
    assert question.answer_status == "suggested"
    assert "review_history" not in question.ocr_raw_json["evidence_bundle"]
