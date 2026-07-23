"""Build clean, printable practice questions from MiniMax recognition data."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


QuestionType = Literal[
    "write_pinyin",
    "write_word",
    "fill_blank",
    "calculation",
    "other",
]
QUESTION_TYPES = {
    "write_pinyin",
    "write_word",
    "fill_blank",
    "calculation",
    "other",
}


class MissingPracticePromptError(ValueError):
    """Raised when a legacy question cannot be safely reconstructed."""

    def __init__(self, count: int = 1):
        self.count = count
        super().__init__(f"{count} question(s) are missing structured practice prompts")


class PrintableQuestion(BaseModel):
    wrong_question_id: str
    instruction: str
    prompt_text: str
    question_type: QuestionType
    display_text: str
    answer: Optional[str] = None


def render_display_text(instruction: str, prompt_text: str, question_type: str) -> str:
    """Render a clean question while keeping answers out of the worksheet."""
    if question_type == "write_pinyin":
        return f"给“{prompt_text}”写拼音：________________"
    if question_type == "write_word":
        return f"根据拼音“{prompt_text}”写词语：________________"
    return f"{instruction}\n{prompt_text}\n________________"


def order_questions(questions, question_ids: list[str]) -> list:
    """Return owned questions in the exact order submitted by the client."""
    by_id = {str(question.id): question for question in questions}
    return [by_id[question_id] for question_id in question_ids if question_id in by_id]


def build_printable_questions(questions) -> list[PrintableQuestion]:
    """Build all questions and report the complete legacy-record count."""
    printable = []
    missing_count = 0
    for question in questions:
        try:
            printable.append(build_printable_question(question))
        except MissingPracticePromptError:
            missing_count += 1
    if missing_count:
        raise MissingPracticePromptError(missing_count)
    return printable


def build_printable_question(question) -> PrintableQuestion:
    """Convert structured recognition fields without exposing the student's answer."""
    raw = question.ocr_raw_json or {}
    instruction = str(raw.get("instruction") or "").strip()
    prompt_text = str(raw.get("prompt_text") or "").strip()
    question_type = str(raw.get("question_type") or "other").strip()
    if not instruction or not prompt_text:
        raise MissingPracticePromptError()
    if question_type not in QUESTION_TYPES:
        question_type = "other"

    display_text = render_display_text(instruction, prompt_text, question_type)

    answer = str(question.ocr_answer).strip() if question.ocr_answer else None
    return PrintableQuestion(
        wrong_question_id=str(question.id),
        instruction=instruction,
        prompt_text=prompt_text,
        question_type=question_type,
        display_text=display_text,
        answer=answer,
    )
