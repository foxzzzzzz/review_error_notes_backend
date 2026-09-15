import ast
from pathlib import Path
from uuid import uuid4

import pytest

from app.schemas.question import QuestionOut, ReviewDecision


def test_question_detail_exposes_structured_review_context():
    source = (Path(__file__).parents[2] / "app" / "schemas" / "question.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    question_out = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QuestionOut"
    )
    fields = {
        node.target.id
        for node in question_out.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert "ocr_answer" in fields
    assert "ocr_raw_json" in fields


def test_review_reprocessing_accepts_force_unmarked_mode():
    from app.schemas.question import ReviewImageReprocessRequest

    assert ReviewImageReprocessRequest(correction="force_unmarked").correction == "force_unmarked"


def test_question_detail_exposes_evidence_statuses():
    fields = QuestionOut.model_fields

    for field in (
        "recognition_pipeline",
        "mark_status",
        "question_evidence_status",
        "answer_status",
    ):
        assert fields[field].annotation == str | None


def test_collect_allows_legacy_request_without_correct_answer():
    decision = ReviewDecision(question_id=uuid4(), decision="collect")

    assert decision.correct_answer is None


def test_ignore_does_not_require_answer():
    decision = ReviewDecision(question_id=uuid4(), decision="ignore")

    assert decision.correct_answer is None


def test_question_type_accepts_database_column_limit():
    decision = ReviewDecision(
        question_id=uuid4(),
        decision="collect",
        question_type="x" * 20,
    )

    assert decision.question_type == "x" * 20


def test_question_type_over_database_column_limit_is_pydantic_invalid():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        ReviewDecision(
            question_id=uuid4(),
            decision="collect",
            question_type="x" * 21,
        )

    assert exc_info.value.errors()[0]["type"] == "string_too_long"


def test_question_type_over_database_column_limit_returns_http_422():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/decision")
    def validate_decision(decision: ReviewDecision):
        return decision

    response = TestClient(app).post(
        "/decision",
        json={
            "question_id": str(uuid4()),
            "decision": "collect",
            "question_type": "x" * 21,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "string_too_long"
