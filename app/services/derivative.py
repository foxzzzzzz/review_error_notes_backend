"""Structured derivative-question generation and validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from app.services.practice_question import (
    PrintableQuestion,
    QuestionType,
    render_display_text,
)


class DerivativeGenerationError(RuntimeError):
    """Raised when requested derivatives cannot be safely generated."""


class DerivativePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    instruction: str
    prompt_text: str
    question_type: QuestionType
    answer: str

    @field_validator("instruction", "prompt_text", "answer")
    @classmethod
    def text_must_not_be_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("derivative fields must not be blank")
        return value


def _identity(question) -> str:
    display_text = render_display_text(
        question.instruction,
        question.prompt_text,
        question.question_type,
    )
    return " ".join(display_text.casefold().split())


async def generate_derivative_variants(
    original: PrintableQuestion,
    difficulty: int,
    target_difficulty: int,
    subject: str,
    count: int,
    llm_generator=None,
) -> list[PrintableQuestion]:
    """Generate validated, non-duplicated practice questions."""
    if count == 0:
        return []
    if llm_generator is None:
        from app.services.llm import generate_derivative as llm_generator

    seen = {_identity(original)}
    variants = []
    for _ in range(count):
        try:
            payload = DerivativePayload.model_validate(
                await llm_generator(
                    original=original,
                    difficulty=difficulty,
                    target_difficulty=target_difficulty,
                    subject=subject,
                )
            )
        except Exception as exc:
            raise DerivativeGenerationError("Derivative generation failed validation") from exc

        identity = _identity(payload)
        if identity in seen:
            raise DerivativeGenerationError("Derivative generation returned a duplicate question")
        seen.add(identity)
        variants.append(
            PrintableQuestion(
                wrong_question_id=original.wrong_question_id,
                instruction=payload.instruction,
                prompt_text=payload.prompt_text,
                question_type=payload.question_type,
                display_text=render_display_text(
                    payload.instruction,
                    payload.prompt_text,
                    payload.question_type,
                ),
                answer=payload.answer,
            )
        )

    return variants
