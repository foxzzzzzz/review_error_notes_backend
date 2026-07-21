import asyncio
from pathlib import Path

import pytest

from app.services import derivative, llm


def test_sheet_configuration_declares_bounded_fields():
    source = (Path(__file__).parents[2] / "app" / "schemas" / "sheet.py").read_text(encoding="utf-8")

    assert "derived_per_original: int = Field(default=1, ge=1, le=3)" in source
    assert "difficulty_boost: int = Field(default=2, ge=1, le=3)" in source


def test_llm_derivative_failure_propagates_to_the_caller(monkeypatch):
    async def fail(_prompt):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(llm, "_call_llm", fail)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(llm.generate_derivative("1 + 1 =", {}, 1, 2, "math"))


def test_empty_llm_derivative_is_treated_as_a_generation_failure(monkeypatch):
    async def empty(_prompt):
        return "   "

    monkeypatch.setattr(llm, "_call_llm", empty)

    with pytest.raises(ValueError, match="empty derivative"):
        asyncio.run(llm.generate_derivative("1 + 1 =", {}, 1, 2, "math"))


def test_generate_variants_honors_count_and_records_llm_method():
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return f"derived-{len(calls)}"

    variants = asyncio.run(derivative.generate_derivative_variants(
        question_text="1 + 1 =",
        problem_schema={},
        difficulty=1,
        target_difficulty=2,
        subject="math",
        count=3,
        use_llm=True,
        llm_generator=generate,
    ))

    assert variants == [
        ("derived-1", "llm"),
        ("derived-2", "llm"),
        ("derived-3", "llm"),
    ]


def test_generate_variants_uses_rule_method_when_llm_fails():
    async def fail(**_kwargs):
        raise RuntimeError("provider unavailable")

    variants = asyncio.run(derivative.generate_derivative_variants(
        question_text="10 + 2 =",
        problem_schema={},
        difficulty=1,
        target_difficulty=2,
        subject="math",
        count=1,
        use_llm=True,
        llm_generator=fail,
    ))

    assert len(variants) == 1
    assert variants[0][1] == "rule"
    assert variants[0][0] != "10 + 2 ="
