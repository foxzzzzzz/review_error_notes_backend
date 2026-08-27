import pytest


@pytest.mark.parametrize(
    (
        "mode",
        "student_answer",
        "reference_answer",
        "unanswered",
        "review_status",
        "reliable_error_mark",
        "local_ocr_status",
        "ocr_enabled",
        "expected_action",
    ),
    [
        ("unmarked", "合作", "合作", False, "confirmed", False, "support", True, "discard"),
        ("unmarked", "合做", "合作", False, "confirmed", False, "support", True, "review"),
        ("unmarked", "", "合作", True, "confirmed", False, "support", True, "review"),
        ("unmarked", "合做", "合作", False, "confirmed", False, "wrong_candidate", True, "discard"),
        ("unmarked", "合做", "合作", False, "confirmed", False, "text_mismatch", True, "discard"),
        ("unmarked", "合做", "合作", False, "confirmed", False, "unavailable", True, "review"),
        ("unmarked", "合做", "合作", False, "confirmed", False, "disabled", False, "review"),
        ("marked", "合做", "合作", False, "confirmed", True, "support", True, "collect"),
        ("marked", "合做", "合作", False, "confirmed", False, "support", True, "discard"),
        ("marked", "合作", "合作", False, "confirmed", True, "support", True, "review"),
        ("marked", "合做", "合作", False, "needs_review", True, "support", True, "review"),
        ("marked", "合做", "合作", False, "confirmed", True, "disabled", False, "collect"),
        ("marked", "合做", "合作", False, "confirmed", True, "wrong_candidate", True, "review"),
        ("marked", "合做", "合作", False, "confirmed", True, "inconclusive", True, "review"),
    ],
)
def test_candidate_decision_table(
    mode,
    student_answer,
    reference_answer,
    unanswered,
    review_status,
    reliable_error_mark,
    local_ocr_status,
    ocr_enabled,
    expected_action,
):
    from app.services.recognition_policy import decide_candidate

    decision = decide_candidate(
        mode=mode,
        student_answer=student_answer,
        reference_answer=reference_answer,
        unanswered=unanswered,
        review_status=review_status,
        reliable_error_mark=reliable_error_mark,
        local_ocr_status=local_ocr_status,
        ocr_enabled=ocr_enabled,
    )

    assert decision.action == expected_action
    assert decision.reason


def test_collection_status_adapter_does_not_persist_discarded_items():
    from app.services.question_collection import collection_status_for_decision
    from app.services.recognition_policy import CandidateDecision

    assert collection_status_for_decision(CandidateDecision(action="collect", reason="x")) == "collected"
    assert collection_status_for_decision(CandidateDecision(action="review", reason="x")) == "pending_review"
    assert collection_status_for_decision(CandidateDecision(action="discard", reason="x")) is None
