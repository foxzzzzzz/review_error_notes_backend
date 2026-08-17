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


class DerivativeBatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_id: str
    original: PrintableQuestion
    difficulty: int
    target_difficulty: int
    subject: str


class DerivativeBatchItemPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_id: str
    variants: list[DerivativePayload]


class DerivativeBatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: list[DerivativeBatchItemPayload]


class DerivativeBatchGenerationResult(BaseModel):
    variants_by_source_id: dict[str, list[PrintableQuestion]]
    usage: dict[str, int]


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


async def generate_derivative_batch(
    items: list[DerivativeBatchInput],
    count: int,
    llm_generator=None,
    client=None,
) -> DerivativeBatchGenerationResult:
    """Generate and validate derivatives for multiple source questions."""
    requested_ids = [item.source_id for item in items]
    if len(set(requested_ids)) != len(requested_ids):
        raise DerivativeGenerationError("Batch input contains duplicate source IDs")
    if count < 0:
        raise DerivativeGenerationError("Derivative count must not be negative")
    if count == 0:
        return DerivativeBatchGenerationResult(
            variants_by_source_id={source_id: [] for source_id in requested_ids},
            usage={},
        )
    if llm_generator is None:
        from app.services.llm import generate_derivative_batch as llm_generator

    try:
        raw_payload, raw_usage = await llm_generator(
            items=items,
            count=count,
            client=client,
        )
        payload = DerivativeBatchPayload.model_validate(raw_payload)
    except Exception as exc:
        raise DerivativeGenerationError("Batch derivative generation failed validation") from exc

    returned_ids = [item.source_id for item in payload.items]
    if len(set(returned_ids)) != len(returned_ids):
        raise DerivativeGenerationError("Batch response contains duplicate source IDs")
    if set(returned_ids) != set(requested_ids):
        raise DerivativeGenerationError("Batch response source IDs do not match request")

    payload_by_id = {item.source_id: item for item in payload.items}
    variants_by_source_id = {}
    for batch_input in items:
        item_payload = payload_by_id[batch_input.source_id]
        if len(item_payload.variants) != count:
            raise DerivativeGenerationError("Batch response derivative count is invalid")

        seen = {_identity(batch_input.original)}
        variants = []
        for derivative in item_payload.variants:
            identity = _identity(derivative)
            if identity in seen:
                raise DerivativeGenerationError(
                    "Batch derivative generation returned a duplicate question"
                )
            seen.add(identity)
            variants.append(
                PrintableQuestion(
                    wrong_question_id=batch_input.original.wrong_question_id,
                    instruction=derivative.instruction,
                    prompt_text=derivative.prompt_text,
                    question_type=derivative.question_type,
                    display_text=render_display_text(
                        derivative.instruction,
                        derivative.prompt_text,
                        derivative.question_type,
                    ),
                    answer=derivative.answer,
                )
            )
        variants_by_source_id[batch_input.source_id] = variants

    usage = {
        str(key): int(value)
        for key, value in dict(raw_usage or {}).items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    return DerivativeBatchGenerationResult(
        variants_by_source_id=variants_by_source_id,
        usage=usage,
    )
