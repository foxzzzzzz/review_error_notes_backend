import asyncio
import json
from pathlib import Path

import httpx
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


def _batch_inputs():
    from app.services.derivative import DerivativeBatchInput

    values = []
    for index, prompt_text in enumerate(("计算", "课文")):
        original = PrintableQuestion(
            wrong_question_id=f"question-{index}",
            instruction="看词语写拼音",
            prompt_text=prompt_text,
            question_type="write_pinyin",
            display_text=f"给“{prompt_text}”写拼音：________________",
            answer="jì suàn" if index == 0 else "kè wén",
        )
        values.append(
            DerivativeBatchInput(
                source_id=f"question-{index}",
                original=original,
                difficulty=1,
                target_difficulty=3,
                subject="chinese",
            )
        )
    return values


def _batch_response():
    return {
        "items": [
            {
                "source_id": "question-1",
                "variants": [_variant("文章", "wén zhāng")],
            },
            {
                "source_id": "question-0",
                "variants": [_variant("算式", "suàn shì")],
            },
        ]
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


def test_generate_batch_maps_provider_response_back_to_request_order():
    async def generate(**_kwargs):
        return _batch_response(), {"prompt_tokens": 120, "completion_tokens": 80}

    result = asyncio.run(
        derivative.generate_derivative_batch(
            items=_batch_inputs(),
            count=1,
            llm_generator=generate,
        )
    )

    assert list(result.variants_by_source_id) == ["question-0", "question-1"]
    assert result.variants_by_source_id["question-0"][0].prompt_text == "算式"
    assert result.variants_by_source_id["question-1"][0].prompt_text == "文章"
    assert result.usage == {"prompt_tokens": 120, "completion_tokens": 80}


@pytest.mark.parametrize(
    "response",
    [
        {"items": [_batch_response()["items"][0]]},
        {**_batch_response(), "unexpected": "field"},
        {
            "items": [
                _batch_response()["items"][0],
                {"source_id": "unknown", "variants": [_variant()]},
            ]
        },
        {
            "items": [
                _batch_response()["items"][0],
                _batch_response()["items"][0],
            ]
        },
        {
            "items": [
                _batch_response()["items"][0],
                {"source_id": "question-0", "variants": []},
            ]
        },
        {
            "items": [
                _batch_response()["items"][0],
                {
                    "source_id": "question-0",
                    "variants": [_variant("计算", "jì suàn")],
                },
            ]
        },
    ],
)
def test_generate_batch_rejects_invalid_id_count_or_original_copy(response):
    async def generate(**_kwargs):
        return response, {}

    with pytest.raises(derivative.DerivativeGenerationError):
        asyncio.run(
            derivative.generate_derivative_batch(
                items=_batch_inputs(),
                count=1,
                llm_generator=generate,
            )
        )


def test_generate_batch_rejects_duplicate_variants():
    response = _batch_response()
    for item in response["items"]:
        item["variants"] = [item["variants"][0], item["variants"][0]]

    async def generate(**_kwargs):
        return response, {}

    with pytest.raises(derivative.DerivativeGenerationError, match="duplicate"):
        asyncio.run(
            derivative.generate_derivative_batch(
                items=_batch_inputs(),
                count=2,
                llm_generator=generate,
            )
        )


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


def test_llm_batch_prompt_contains_multiple_ids_and_returns_usage(monkeypatch):
    captured = {}
    response = _batch_response()
    shared_client = object()

    async def respond(prompt, client=None):
        captured["prompt"] = prompt
        captured["client"] = client
        return type(
            "CallResult",
            (),
            {
                "content": json.dumps(response, ensure_ascii=False),
                "usage": {"prompt_tokens": 120, "completion_tokens": 80},
            },
        )()

    monkeypatch.setattr(llm, "_call_llm_with_usage", respond, raising=False)

    payload, usage = asyncio.run(
        llm.generate_derivative_batch(
            items=_batch_inputs(),
            count=1,
            client=shared_client,
        )
    )

    assert payload == response
    assert usage == {"prompt_tokens": 120, "completion_tokens": 80}
    assert captured["client"] is shared_client
    assert "question-0" in captured["prompt"]
    assert "question-1" in captured["prompt"]
    assert '"derived_count": 1' in captured["prompt"]
    assert "student_name" not in captured["prompt"]


def test_llm_call_reuses_supplied_client_and_extracts_numeric_usage():
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": "<output>{}</output>"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "details": {"ignored": 1},
                },
            }

    class Client:
        def __init__(self):
            self.calls = []

        async def post(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return Response()

    client = Client()
    result = asyncio.run(llm._call_llm_with_usage("prompt", client=client))

    assert result.content == "{}"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert len(client.calls) == 1


def test_llm_call_retries_transport_failure_with_the_supplied_client(monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": "<output>{}</output>"}}],
                "usage": {},
            }

    class Client:
        def __init__(self):
            self.call_count = 0

        async def post(self, *_args, **_kwargs):
            self.call_count += 1
            if self.call_count == 1:
                raise httpx.ConnectError("temporary failure")
            return Response()

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = Client()

    result = asyncio.run(llm._call_llm_with_usage("prompt", client=client))

    assert result.content == "{}"
    assert client.call_count == 2


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
