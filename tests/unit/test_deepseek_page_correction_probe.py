import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image


def _settings():
    return SimpleNamespace(
        DEEPSEEK_VISION_KEY_SOURCE="dedicated",
        DEEPSEEK_VISION_API_KEY="secret-not-for-artifacts",
        DEEPSEEK_VISION_API_BASE="https://api.deepseek.test/v1",
        DEEPSEEK_VISION_MODEL="deepseek-flash",
        DEEPSEEK_VISION_THINKING="disabled",
        DEEPSEEK_VISION_MAX_TOKENS=4096,
        DEEPSEEK_VISION_IMAGE_DETAIL="original",
        DEEPSEEK_VISION_TIMEOUT_SECONDS=60,
        MINIMAX_IMAGE_MAX_EDGE=2048,
        MINIMAX_IMAGE_JPEG_QUALITY=90,
        LLM_API_KEY="",
        LLM_API_BASE="https://api.deepseek.test/v1",
    )


def _inputs(tmp_path):
    image = tmp_path / "P010.jpg"
    Image.new("RGB", (100, 150), "white").save(image)
    first = tmp_path / "first.md"
    first.write_text('```json\n' + json.dumps({"wrong_questions": [
        {"printed_question": "zú qiú", "student_answer": "足球",
         "model_bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9,
         "uncertain_fields": []},
        {"printed_question": "bīng kuài", "student_answer": "冰快",
         "model_bbox": [0.3, 0.2, 0.5, 0.4], "confidence": 0.9,
         "uncertain_fields": []},
    ]}, ensure_ascii=False) + '\n```', encoding="utf-8")
    prompt = tmp_path / "correction.md"
    prompt.write_text("逐项核验红圈或红叉；无明确证据时剔除。", encoding="utf-8")
    return image, first, prompt


def test_p010_probe_sends_first_result_and_image_once_without_changing_pipeline(tmp_path):
    from scripts.deepseek_page_correction_probe import run_probe

    image, first, prompt = _inputs(tmp_path)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "deepseek-flash",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "decisions": [
                    {"candidate_id": 0, "verdict": "reject", "mark_type": None,
                     "mark_bbox": None},
                    {"candidate_id": 1, "verdict": "keep", "mark_type": "red_cross",
                     "mark_bbox": [0.31, 0.21, 0.36, 0.27]},
                ]
            })}}],
        })

    output = tmp_path / "probe"
    result = run_probe(
        image_path=image, first_response_path=first, prompt_path=prompt,
        output_dir=output, settings_obj=_settings(),
        transport=httpx.MockTransport(respond),
    )

    assert len(requests) == 1
    body = requests[0]
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "disabled"}
    assert len(body["messages"]) == 1
    content = body["messages"][0]["content"]
    assert '"candidate_id": 0' in content[0]["text"]
    assert '"printed_question": "bīng kuài"' in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    sent_image = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
    assert (output / "input.jpg").read_bytes() == sent_image
    assert result["first_candidate_count"] == 2
    assert result["kept_candidate_ids"] == [1]
    assert result["kept_candidates"][0]["printed_question"] == "bīng kuài"
    assert json.loads((output / "decision.json").read_text(encoding="utf-8")) == result
    artifacts = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir()
        if path.suffix != ".jpg"
    )
    assert "secret-not-for-artifacts" not in artifacts
    assert "base64," not in artifacts


def test_p010_probe_rejects_incomplete_decision_set_but_keeps_raw_response(tmp_path):
    from scripts.deepseek_page_correction_probe import run_probe

    image, first, prompt = _inputs(tmp_path)
    response = {"model": "deepseek-flash", "choices": [{"finish_reason": "stop",
        "message": {"content": '{"decisions":[{"candidate_id":1,"verdict":"keep","mark_type":"red_cross","mark_bbox":[0.31,0.21,0.36,0.27]}]}'}}]}

    with pytest.raises(ValueError, match="one decision per input candidate"):
        run_probe(
            image_path=image, first_response_path=first, prompt_path=prompt,
            output_dir=tmp_path / "probe", settings_obj=_settings(),
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
        )

    assert json.loads((tmp_path / "probe" / "response.json").read_text(encoding="utf-8")) == response


def test_probe_accepts_explanation_before_single_json_block(tmp_path):
    from scripts.deepseek_page_correction_probe import run_probe

    image, first, prompt = _inputs(tmp_path)
    content = (
        "逐项复核：只有冰快附近有红圈。\n"
        "```json\n"
        '{"decisions":['
        '{"candidate_id":0,"verdict":"reject","mark_type":null,"mark_bbox":null},'
        '{"candidate_id":1,"verdict":"keep","mark_type":"red_circle",'
        '"mark_bbox":[0.18,0.65,0.29,0.74]}]}'
        "\n```\n"
    )
    response = {"model": "deepseek-flash", "choices": [{
        "finish_reason": "stop", "message": {"content": content},
    }]}

    result = run_probe(
        image_path=image, first_response_path=first, prompt_path=prompt,
        output_dir=tmp_path / "probe", settings_obj=_settings(),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )

    assert result["kept_candidate_ids"] == [1]
    assert result["decisions"][1]["mark_bbox"] == [0.18, 0.65, 0.29, 0.74]


@pytest.mark.parametrize("content", [
    "逐项复核：候选 1 有红圈。",
    '说明\n```json\n{"decisions":[]}\n```\n另一个结果\n```json\n{"decisions":[]}\n```',
])
def test_probe_rejects_unstructured_or_multiple_json_blocks(tmp_path, content):
    from scripts.deepseek_page_correction_probe import run_probe

    image, first, prompt = _inputs(tmp_path)
    response = {"model": "deepseek-flash", "choices": [{
        "finish_reason": "stop", "message": {"content": content},
    }]}

    with pytest.raises(ValueError, match="one JSON code block"):
        run_probe(
            image_path=image, first_response_path=first, prompt_path=prompt,
            output_dir=tmp_path / "probe", settings_obj=_settings(),
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
        )
    assert (tmp_path / "probe" / "answer.md").read_text(encoding="utf-8").strip() == content


def test_probe_extracts_single_trailing_bare_json_object():
    from scripts.deepseek_page_correction_probe import _extract_correction_json

    content = (
        "逐项复核：只有候选 13 有明确红圈与红叉。\n\n"
        '{"decisions":[{"candidate_id":13,"verdict":"keep",'
        '"mark_type":"red_circle_and_cross",'
        '"mark_bbox":[0.186,0.664,0.281,0.734]}]}'
    )

    assert _extract_correction_json(content) == {"decisions": [
        {"candidate_id": 13, "verdict": "keep",
         "mark_type": "red_circle_and_cross",
         "mark_bbox": [0.186, 0.664, 0.281, 0.734]},
    ]}


@pytest.mark.parametrize("content", [
    '说明 {"other":true}\n{"decisions":[]}',
    '{"decisions":[]}\n补充说明',
    '说明\n{"decisions":[',
    '只有说明文字，没有结构化结果。',
])
def test_probe_rejects_invalid_trailing_bare_json(content):
    from scripts.deepseek_page_correction_probe import _extract_correction_json

    with pytest.raises(ValueError, match="one JSON code block"):
        _extract_correction_json(content)
