import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx


ROOT = Path(__file__).resolve().parents[2]
PROMPT = (
    "图片为错题图，其中错题被人为用红笔做了批改：红圈和红叉标记了错题，"
    "每一个红圈+红叉对应一个错题。\n\n"
    "你基于上面的信息，把错题区域和内容识别出来。"
)


def _settings():
    return SimpleNamespace(
        DEEPSEEK_VISION_KEY_SOURCE="text_llm",
        DEEPSEEK_VISION_API_KEY="",
        DEEPSEEK_VISION_API_BASE="https://api.deepseek.test/v1",
        DEEPSEEK_VISION_MODEL="deepseek-flash",
        DEEPSEEK_VISION_THINKING="disabled",
        DEEPSEEK_VISION_MAX_TOKENS=4096,
        DEEPSEEK_VISION_IMAGE_DETAIL="original",
        DEEPSEEK_VISION_TIMEOUT_SECONDS=60,
        LLM_API_KEY="secret-key-that-must-not-be-written",
        LLM_API_BASE="https://api.deepseek.test/v1",
    )


def test_raw_comparison_sends_each_original_once_and_preserves_free_text(tmp_path):
    from scripts.deepseek_raw_page_comparison import run_comparison

    prompt_path = ROOT / "config" / "deepseek-raw-page-comparison-prompt.md"
    assert prompt_path.read_text(encoding="utf-8").strip() == PROMPT
    pages = []
    original_bytes = {}
    for index, (label, truth_count, suffix) in enumerate(
        [("P003", 11, ".jpg"), ("P015", 9, ".png"), ("P041", 6, ".webp")]
    ):
        image_path = tmp_path / f"source-{index}{suffix}"
        image_bytes = f"original-image-{label}".encode()
        image_path.write_bytes(image_bytes)
        original_bytes[label] = image_bytes
        pages.append(
            {
                "label": label,
                "image_path": image_path,
                "source_name": f"{label}{suffix}",
                "truth_count": truth_count,
            }
        )

    requests = []

    def respond(request: httpx.Request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash",
                "choices": [
                    {
                        "message": {"content": f"自然语言识别结果-{len(requests)}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 17},
            },
        )

    output_dir = tmp_path / "raw-direct"
    summary = run_comparison(
        pages=pages,
        prompt_path=prompt_path,
        output_dir=output_dir,
        settings_obj=_settings(),
        transport=httpx.MockTransport(respond),
    )

    assert len(requests) == 3
    assert [row["status"] for row in summary] == ["completed"] * 3
    for index, (page, request_body) in enumerate(zip(pages, requests), start=1):
        assert set(request_body) == {"model", "thinking", "max_tokens", "messages"}
        assert request_body["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/{'jpeg' if page['source_name'].endswith('.jpg') else page['source_name'].rsplit('.', 1)[1]};base64,"
                                + base64.b64encode(original_bytes[page["label"]]).decode()
                            ),
                            "detail": "original",
                        },
                    },
                ],
            }
        ]
        page_dir = output_dir / page["label"]
        assert (page_dir / "answer.md").read_text(encoding="utf-8") == (
            f"自然语言识别结果-{index}\n"
        )
        response = json.loads((page_dir / "response.json").read_text(encoding="utf-8"))
        assert response["choices"][0]["message"]["content"] == f"自然语言识别结果-{index}"

    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    assert "secret-key-that-must-not-be-written" not in persisted
    assert "base64," not in persisted
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "summary.md").is_file()


def test_raw_comparison_records_failures_without_retrying_or_stopping_other_pages(tmp_path):
    from scripts.deepseek_raw_page_comparison import run_comparison

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text(PROMPT, encoding="utf-8")
    pages = []
    for label in ("P003", "P015", "P041"):
        image_path = tmp_path / f"{label}.jpg"
        image_path.write_bytes(label.encode())
        pages.append(
            {
                "label": label,
                "image_path": image_path,
                "source_name": image_path.name,
                "truth_count": None,
            }
        )
    calls = 0

    def respond(_request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="temporary upstream failure")

    summary = run_comparison(
        pages=pages,
        prompt_path=prompt_path,
        output_dir=tmp_path / "raw-direct",
        settings_obj=_settings(),
        transport=httpx.MockTransport(respond),
    )

    assert calls == 3
    assert [row["status"] for row in summary] == ["failed"] * 3
    assert all(row["http_status"] == 503 for row in summary)
