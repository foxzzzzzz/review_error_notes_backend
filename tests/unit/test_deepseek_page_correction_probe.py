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
