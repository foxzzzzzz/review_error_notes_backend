"""Pure recognition-mode candidate collection policy."""

from __future__ import annotations

import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict


RecognitionMode = Literal["marked", "unmarked"]
OCRStatus = Literal[
    "support",
    "wrong_candidate",
    "text_mismatch",
    "inconclusive",
    "unavailable",
    "disabled",
]


class CandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["collect", "review", "discard"]
    reason: str


def _normalized_answer(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] in {"L", "M", "N"}
    )


def decide_candidate(
    *,
    mode: RecognitionMode,
    student_answer: str | None,
    reference_answer: str | None,
    unanswered: bool,
    review_status: str,
    reliable_error_mark: bool,
    local_ocr_status: OCRStatus,
    ocr_enabled: bool,
) -> CandidateDecision:
    """Return an explicit persistence action from bounded evidence."""
    if local_ocr_status in {"wrong_candidate", "text_mismatch"}:
        return CandidateDecision(
            action="discard" if mode == "unmarked" else "review",
            reason="本地 OCR 无法支持当前候选题定位",
        )

    if mode == "marked" and not reliable_error_mark:
        return CandidateDecision(
            action="discard",
            reason="红标作业中未关联到可靠错误标记",
        )

    student = _normalized_answer(student_answer)
    reference = _normalized_answer(reference_answer)
    answers_match = bool(student and reference and student == reference)
    if mode == "unmarked" and answers_match:
        return CandidateDecision(
            action="discard",
            reason="学生答案与参考答案一致",
        )

    if ocr_enabled and local_ocr_status == "unavailable":
        return CandidateDecision(
            action="review",
            reason="本地 OCR 暂不可用，需要人工确认",
        )

    uncertain = review_status != "confirmed"
    missing_comparable_answer = unanswered or not student or not reference
    if mode == "marked":
        if uncertain or missing_comparable_answer or answers_match:
            return CandidateDecision(
                action="review",
                reason="红标、题目定位或答案证据需要人工确认",
            )
        if local_ocr_status == "support" or (
            not ocr_enabled and local_ocr_status == "disabled"
        ):
            return CandidateDecision(
                action="collect",
                reason="可靠红标及答案不一致证据得到支持",
            )
        return CandidateDecision(
            action="review",
            reason="本地 OCR 证据不足，需要人工确认",
        )

    return CandidateDecision(
        action="review",
        reason=(
            "题目未作答或缺少可比较答案"
            if missing_comparable_answer
            else "无红标作业中检测到可信答案差异"
        ),
    )
