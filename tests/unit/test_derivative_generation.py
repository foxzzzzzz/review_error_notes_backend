import asyncio
import json
from pathlib import Path

import pytest

from app.services import derivative, llm
from app.services.practice_question import PrintableQuestion


def _original():
    return PrintableQuestion(
        wrong_question_id="question-id",
        instruction="看词语写拼音",
        prompt_text="计算",
        question_type="write_pinyin",
        display_text="给“计算”写拼音：________________",
        answer="jì suàn",
    )


def _variant(prompt_text="算式", answer="suàn shì"):
    return {
        "instruction": "看词语写拼音",
        "prompt_text": prompt_text,
        "question_type": "write_pinyin",
        "answer": answer,
    }


def test_sheet_configuration_defaults_to_originals_only():
    source = (Path(__file__).parents[2] / "app" / "schemas" / "sheet.py").read_text(encoding="utf-8")

    assert "derived_per_original: int = Field(default=0, ge=0, le=3)" in source


def test_zero_count_does_not_call_generator():
    async def fail(**_kwargs):
        raise AssertionError("generator must not be called")

    variants = asyncio.run(
        derivative.generate_derivative_variants(
            original=_original(),
            difficulty=1,
            target_difficulty=2,
            subject="chinese",
            count=0,
            llm_generator=fail,
        )
    )

    assert variants == []


def test_generate_variants_returns_structured_unique_questions():
    values = [_variant("算式", "suàn shì"), _variant("课文", "kè wén")]

    async def generate(**_kwargs):
        return values.pop(0)

    variants = asyncio.run(
        derivative.generate_derivative_variants(
            original=_original(),
            difficulty=1,
            target_difficulty=2,
            subject="chinese",
            count=2,
            llm_generator=generate,
        )
    )

    assert [item.prompt_text for item in variants] == ["算式", "课文"]
    assert [item.display_text for item in variants] == [
        "给“算式”写拼音：________________",
        "给“课文”写拼音：________________",
    ]


@pytest.mark.parametrize(
    "generated",
    [
        _variant("计算", "jì suàn"),
        {
            "instruction": "请给下面词语标注拼音",
            "prompt_text": "计算",
            "question_type": "write_pinyin",
            "answer": "jì suàn",
        },
        {"instruction": "", "prompt_text": "算式", "question_type": "write_pinyin", "answer": "suàn shì"},
    ],
)
def test_invalid_or_original_copy_fails_without_rule_fallback(generated):
    async def generate(**_kwargs):
        return generated

    with pytest.raises(derivative.DerivativeGenerationError):
        asyncio.run(
            derivative.generate_derivative_variants(
                original=_original(),
                difficulty=1,
                target_difficulty=2,
                subject="chinese",
                count=1,
                llm_generator=generate,
            )
        )


def test_provider_transport_failure_is_mapped_to_generation_error():
    async def generate(**_kwargs):
        raise OSError("network unavailable")

    with pytest.raises(derivative.DerivativeGenerationError):
        asyncio.run(
            derivative.generate_derivative_variants(
                original=_original(),
                difficulty=1,
                target_difficulty=2,
                subject="chinese",
                count=1,
                llm_generator=generate,
            )
        )


def test_duplicate_derivatives_are_rejected():
    async def generate(**_kwargs):
        return _variant()

    with pytest.raises(derivative.DerivativeGenerationError, match="duplicate"):
        asyncio.run(
            derivative.generate_derivative_variants(
                original=_original(),
                difficulty=1,
                target_difficulty=2,
                subject="chinese",
                count=2,
                llm_generator=generate,
            )
        )


def test_same_printed_question_is_rejected_even_if_question_type_changes():
    original = PrintableQuestion(
        wrong_question_id="question-id",
        instruction="计算下面各题",
        prompt_text="1 + 1 =",
        question_type="calculation",
        display_text="计算下面各题\n1 + 1 =\n________________",
        answer="2",
    )

    async def generate(**_kwargs):
        return {
            "instruction": "计算下面各题",
            "prompt_text": "1 + 1 =",
            "question_type": "other",
            "answer": "2",
        }

    with pytest.raises(derivative.DerivativeGenerationError, match="duplicate"):
        asyncio.run(
            derivative.generate_derivative_variants(
                original=original,
                difficulty=1,
                target_difficulty=2,
                subject="math",
                count=1,
                llm_generator=generate,
            )
        )


def test_llm_derivative_prompt_requests_structured_json(monkeypatch):
    captured = {}

    async def respond(prompt):
        captured["prompt"] = prompt
        return json.dumps(_variant(), ensure_ascii=False)

    monkeypatch.setattr(llm, "_call_llm", respond)

    result = asyncio.run(
        llm.generate_derivative(
            original=_original(),
            difficulty=1,
            target_difficulty=2,
            subject="chinese",
        )
    )

    assert result == _variant()
    assert "instruction" in captured["prompt"]
    assert "prompt_text" in captured["prompt"]
    assert "answer" in captured["prompt"]
    assert "shǔan" not in captured["prompt"]


def test_llm_derivative_failure_propagates_to_the_caller(monkeypatch):
    async def fail(_prompt):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(llm, "_call_llm", fail)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            llm.generate_derivative(
                original=_original(),
                difficulty=1,
                target_difficulty=2,
                subject="chinese",
            )
        )
