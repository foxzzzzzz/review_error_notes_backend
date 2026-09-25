import base64
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image


def _settings():
    return SimpleNamespace(
        DEEPSEEK_VISION_KEY_SOURCE="dedicated",
        DEEPSEEK_VISION_API_KEY="benchmark-secret",
        DEEPSEEK_VISION_API_BASE="https://api.deepseek.test/v1",
        DEEPSEEK_VISION_MODEL="deepseek-flash",
        DEEPSEEK_VISION_THINKING="disabled",
        DEEPSEEK_VISION_MAX_TOKENS=4096,
        DEEPSEEK_VISION_IMAGE_DETAIL="original",
        DEEPSEEK_VISION_TIMEOUT_SECONDS=60,
        MINIMAX_IMAGE_MAX_EDGE=2048,
        MINIMAX_IMAGE_JPEG_QUALITY=90,
        CHINESE_DEEPSEEK_PAGE_PROMPT_PATH="config/deepseek-answer-strategy-detection-prompt.md",
        LLM_API_KEY="",
        LLM_API_BASE="https://api.deepseek.test/v1",
    )


def _candidate():
    return {"printed_question": "看图写词", "student_answer": "冰快",
            "model_bbox": [0.1, 0.2, 0.4, 0.5], "confidence": 0.9,
            "uncertain_fields": []}


def _response(content):
    return httpx.Response(200, json={"model": "deepseek-flash", "usage": {"total_tokens": 12},
        "choices": [{"finish_reason": "stop", "message": {"content": content}}]})


def test_a_and_b_use_identical_prepared_bytes_save_review_artifacts(tmp_path):
    from scripts.deepseek_answer_strategy_benchmark import run_benchmark

    image = tmp_path / "P010.png"
    Image.new("RGB", (120, 180), "white").save(image)
    candidate = _candidate()
    requests = []

    def respond(request):
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        requests.append((content[0]["text"], base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])))
        prompt = content[0]["text"]
        if "候选列表：" in prompt:
            return _response('{"answers":[{"candidate_id":0,"correct_answer_suggestion":"冰块"}]}')
        if "correct_answer_suggestion" in prompt:
            return _response(json.dumps({"wrong_questions": [{**candidate, "correct_answer_suggestion": "冰块"}]}, ensure_ascii=False))
        return _response(json.dumps({"wrong_questions": [candidate]}, ensure_ascii=False))

    out = tmp_path / "out"
    records = run_benchmark(pages=[{"label": "P010", "image_path": str(image)}], output_dir=out,
                            repeats=1, settings_obj=_settings(), transport=httpx.MockTransport(respond))

    assert len(requests) == 3  # A one call; B detection plus one batch call
    assert all(image_bytes == requests[0][1] for _, image_bytes in requests)
    assert len({image_bytes for _, image_bytes in requests}) == 1
    assert records[0]["strategies"]["B"]["first_stage_candidates"] == [candidate]
    assert records[0]["strategies"]["B"]["calls"][-1]["candidate_results"][0]["correct_answer_suggestion"] == "冰块"
    assert (out / "P010/repeat-01/A-review-overlay.jpg").is_file()
    assert (out / "P010/repeat-01/B-review-overlay.jpg").is_file()
    review = (out / "human-review.csv").read_text(encoding="utf-8-sig")
    assert "truth" in review and "candidate" in review and "冰块" in review
    artifacts = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in out.rglob("*.json"))
    assert "benchmark-secret" not in artifacts


def test_production_json_fence_is_accepted_and_raw_response_is_kept(tmp_path):
    from scripts.deepseek_answer_strategy_benchmark import run_benchmark

    image = tmp_path / "P003.jpg"
    Image.new("RGB", (80, 80), "white").save(image)

    def respond(request):
        return _response('```json\n{"wrong_questions":[]}\n```')

    out = tmp_path / "invalid"
    records = run_benchmark(pages=[{"label": "P003", "image_path": str(image)}], output_dir=out,
                            repeats=1, settings_obj=_settings(), transport=httpx.MockTransport(respond))
    a = records[0]["strategies"]["A"]["calls"][0]
    assert a["format_status"] == "valid"
    assert (out / "P003/repeat-01/A-whole-page-raw.md").is_file()
    assert records[0]["strategies"]["B"]["second_stage_skipped"] is True


def test_missing_answer_suggestion_does_not_erase_detected_candidate():
    from scripts.deepseek_answer_strategy_benchmark import _validate_candidates

    candidate = _candidate()
    parsed = _validate_candidates({"wrong_questions": [candidate]}, require_suggestion=True)
    assert parsed["candidates"][0]["correct_answer_suggestion"] is None
    assert parsed["answer_field_missing_count"] == 1


def test_uncertain_bbox_matches_existing_primary_prompt_schema():
    from scripts.deepseek_answer_strategy_benchmark import _validate_candidates

    candidate = {**_candidate(), "uncertain_fields": ["model_bbox"]}
    assert _validate_candidates({"wrong_questions": [candidate]})["candidates"] == [candidate]


def test_invalid_candidate_is_dropped_without_erasing_valid_candidate():
    from scripts.deepseek_answer_strategy_benchmark import _validate_candidates

    valid = _candidate()
    invalid = {**_candidate(), "model_bbox": [0.8, 0.1, 0.2, 0.4]}
    parsed = _validate_candidates({"wrong_questions": [valid, invalid]})
    assert parsed["candidates"] == [valid]
    assert parsed["invalid_item_count"] == 1


def test_b_batch_runs_for_valid_candidates_when_another_candidate_is_invalid(tmp_path):
    from scripts.deepseek_answer_strategy_benchmark import run_benchmark

    image = tmp_path / "mixed.jpg"
    Image.new("RGB", (80, 80), "white").save(image)
    valid = _candidate()
    invalid = {**_candidate(), "model_bbox": [0.8, 0.1, 0.2, 0.4]}
    prompts = []

    def respond(request):
        prompt = json.loads(request.content)["messages"][0]["content"][0]["text"]
        prompts.append(prompt)
        if "候选列表：" in prompt:
            return _response('{"answers":[{"candidate_id":0,"correct_answer_suggestion":"冰块"}]}')
        items = [{**valid, "correct_answer_suggestion": "冰块"}] if "correct_answer_suggestion" in prompt else [valid, invalid]
        return _response(json.dumps({"wrong_questions": items}, ensure_ascii=False))

    records = run_benchmark(
        pages=[{"label": "mixed", "image_path": str(image)}],
        output_dir=tmp_path / "out", repeats=1, settings_obj=_settings(),
        transport=httpx.MockTransport(respond),
    )
    assert len(prompts) == 3
    b = records[0]["strategies"]["B"]
    assert len(b["first_stage_candidates"]) == 1
    assert b["calls"][0]["parsed"]["invalid_item_count"] == 1
    assert b["calls"][1]["candidate_results"][0]["correct_answer_suggestion"] == "冰块"


def test_malformed_batch_answer_keeps_first_stage_and_raw_text(tmp_path):
    from scripts.deepseek_answer_strategy_benchmark import run_benchmark

    image = tmp_path / "malformed.jpg"
    Image.new("RGB", (80, 80), "white").save(image)
    candidate = _candidate()

    def respond(request):
        prompt = json.loads(request.content)["messages"][0]["content"][0]["text"]
        if "候选列表：" in prompt:
            return _response("not json")
        item = {**candidate, "correct_answer_suggestion": None} if "correct_answer_suggestion" in prompt else candidate
        return _response(json.dumps({"wrong_questions": [item]}, ensure_ascii=False))

    out = tmp_path / "out"
    records = run_benchmark(
        pages=[{"label": "malformed", "image_path": str(image)}],
        output_dir=out, repeats=1, settings_obj=_settings(),
        transport=httpx.MockTransport(respond),
    )
    b = records[0]["strategies"]["B"]
    assert len(b["first_stage_candidates"]) == 1
    assert b["calls"][1]["format_status"] == "invalid"
    assert b["calls"][1]["missing_candidate_ids"] == [0]
    assert (out / "malformed/repeat-01/B-answer-batch-raw.md").read_text() == "not json"


def test_partial_batch_answer_keeps_valid_suggestion_for_review(tmp_path):
    from scripts.deepseek_answer_strategy_benchmark import run_benchmark

    image = tmp_path / "partial.jpg"
    Image.new("RGB", (80, 80), "white").save(image)
    first = _candidate()
    second = {**_candidate(), "model_bbox": [0.5, 0.2, 0.8, 0.5]}

    def respond(request):
        prompt = json.loads(request.content)["messages"][0]["content"][0]["text"]
        if "候选列表：" in prompt:
            return _response('{"answers":[{"candidate_id":0,"correct_answer_suggestion":"冰块"}]}')
        items = ([{**first, "correct_answer_suggestion": "冰块"},
                  {**second, "correct_answer_suggestion": None}]
                 if "correct_answer_suggestion" in prompt else [first, second])
        return _response(json.dumps({"wrong_questions": items}, ensure_ascii=False))

    out = tmp_path / "out"
    records = run_benchmark(
        pages=[{"label": "partial", "image_path": str(image)}],
        output_dir=out, repeats=1, settings_obj=_settings(),
        transport=httpx.MockTransport(respond),
    )
    batch = records[0]["strategies"]["B"]["calls"][1]
    assert batch["format_status"] == "partial"
    assert batch["missing_candidate_ids"] == [1]
    assert batch["candidate_results"][0]["correct_answer_suggestion"] == "冰块"
    assert batch["candidate_results"][1]["correct_answer_suggestion"] is None
    assert "冰块" in (out / "human-review.csv").read_text(encoding="utf-8-sig")


def test_truncated_response_still_saves_raw_text(tmp_path):
    from scripts.deepseek_answer_strategy_benchmark import run_benchmark

    image = tmp_path / "truncated.jpg"
    Image.new("RGB", (80, 80), "white").save(image)

    def respond(request):
        return httpx.Response(200, json={"model": "deepseek-flash", "usage": {},
            "choices": [{"finish_reason": "length", "message": {"content": "partial text"}}]})

    out = tmp_path / "out"
    records = run_benchmark(
        pages=[{"label": "truncated", "image_path": str(image)}],
        output_dir=out, repeats=1, settings_obj=_settings(),
        transport=httpx.MockTransport(respond),
    )
    assert (out / "truncated/repeat-01/A-whole-page-raw.md").read_text() == "partial text"
    assert records[0]["strategies"]["B"]["second_stage_skipped"] is True


def test_batch_answer_requires_exact_answer_field():
    from scripts.deepseek_answer_strategy_benchmark import _validate_answers

    parsed = _validate_answers({"answers": [{"candidate_id": 0}]}, [_candidate()])
    assert parsed["answers"] == []
    assert parsed["missing_candidate_ids"] == [0]
    assert parsed["complete"] is False


def test_batch_answer_keeps_unique_valid_rows_and_rejects_both_duplicates():
    from scripts.deepseek_answer_strategy_benchmark import _validate_answers

    parsed = _validate_answers({"answers": [
        {"candidate_id": 0, "correct_answer_suggestion": "冰块"},
        {"candidate_id": 1, "correct_answer_suggestion": "错误一"},
        {"candidate_id": 1, "correct_answer_suggestion": "错误二"},
    ]}, [_candidate(), _candidate(), _candidate()])
    assert parsed["answers"] == [{"candidate_id": 0, "correct_answer_suggestion": "冰块"}]
    assert parsed["missing_candidate_ids"] == [1, 2]
    assert parsed["duplicate_candidate_ids"] == [1]
    assert parsed["complete"] is False


def test_zero_candidates_skip_batch_answer_call(tmp_path):
    from scripts.deepseek_answer_strategy_benchmark import run_benchmark

    image = tmp_path / "blank.jpg"
    Image.new("RGB", (80, 80), "white").save(image)
    requests = []

    def respond(request):
        requests.append(request)
        return _response('{"wrong_questions":[]}')

    records = run_benchmark(
        pages=[{"label": "blank", "image_path": str(image)}],
        output_dir=tmp_path / "out", repeats=1, settings_obj=_settings(),
        transport=httpx.MockTransport(respond),
    )
    assert len(requests) == 2
    assert records[0]["strategies"]["B"]["second_stage_skipped"] is True
