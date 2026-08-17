"""Decide whether a recognized item belongs in the wrong-question library."""

import unicodedata
from typing import Optional, Tuple


COLLECTION_PENDING_REVIEW = "pending_review"
COLLECTION_COLLECTED = "collected"
COLLECTION_IGNORED = "ignored"


def _normalized_answer(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] in {"L", "M", "N"}
    )


def _collection_decision_for(question) -> Tuple[str, str]:
    """Return a conservative collection decision and its user-facing reason."""
    has_reliable_error_mark = bool(
        getattr(question, "reliable_error_mark", False)
    )
    student_answer = _normalized_answer(question.ocr_text)
    reference_answer = _normalized_answer(question.ocr_answer)
    answers_match = bool(student_answer and reference_answer) and (
        student_answer == reference_answer
    )

    if answers_match and not has_reliable_error_mark:
        return COLLECTION_IGNORED, "答案与参考答案一致，未发现可靠错误标记"
    if question.review_status != "confirmed":
        return COLLECTION_PENDING_REVIEW, "题目区域或识别结果不确定"
    if not student_answer or not reference_answer:
        return COLLECTION_PENDING_REVIEW, "缺少可比较的学生作答或参考答案"
    if (
        has_reliable_error_mark
        and not answers_match
    ):
        return COLLECTION_COLLECTED, "检测到可靠错误标记且答案不一致"
    if answers_match:
        return COLLECTION_PENDING_REVIEW, "答案与错误标记不一致"
    return COLLECTION_PENDING_REVIEW, "未检测到可确认的错误标记"


def collection_status_for(question) -> str:
    """Return the collection decision for one recognized item."""
    return _collection_decision_for(question)[0]


def collection_reason_for(question) -> str:
    """Return the reason shown to users for a collection decision."""
    return _collection_decision_for(question)[1]
