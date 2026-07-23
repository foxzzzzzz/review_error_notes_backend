import json

import httpx
import pytest
from PIL import Image


def _write_image(path):
    Image.new("RGB", (400, 300), "white").save(path, format="JPEG")


def _vision_item(raw_text, prompt_text, question_type):
    from app.services.vision_recognition import VisionItem

    return VisionItem(
        raw_text=raw_text,
        instruction="完成练习",
        prompt_text=prompt_text,
        normalized_text=raw_text,
        answer=raw_text,
        subject="chinese",
        question_type=question_type,
        tags=["拼音"],
        difficulty=2,
        confidence=0.95,
        uncertain_segments=[],
        bbox=[0.1, 0.1, 0.2, 0.2],
    )


def _client(handler):
    from app.services.vision_recognition import MiniMaxVisionClient

    return MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )


def test_localization_prompt_defines_the_complete_independent_unit():
    from app.services.vision_recognition import LOCALIZATION_PROMPT, RECOGNITION_PROMPT

    assert "印刷提示、学生答案和相关红色批改标记" in LOCALIZATION_PROMPT
    assert "未标记的相邻兄弟小题" in LOCALIZATION_PROMPT
    assert "每个 index 恰好返回一次" in LOCALIZATION_PROMPT
    assert "tags 只能使用中文标签" in RECOGNITION_PROMPT


def test_localize_sends_all_recognized_indexes_in_one_request(tmp_path):
    source = tmp_path / "question.jpg"
    _write_image(source)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "content": json.dumps(
                    {
                        "items": [
                            {"index": 0, "bbox": [0.1, 0.2, 0.4, 0.5], "confidence": 0.94},
                            {"index": 1, "bbox": [0.5, 0.2, 0.8, 0.5], "confidence": 0.91},
                        ]
                    }
                ),
                "base_resp": {"status_code": 0},
            },
        )

    items = [
        _vision_item("kè wén", "课文", "write_pinyin"),
        _vision_item("合作", "hé zuò", "write_word"),
    ]

    result = _client(handler).localize(str(source), items)

    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["image_url"].startswith("data:image/jpeg;base64,")
    assert '"index": 0' in body["prompt"]
    assert '"index": 1' in body["prompt"]
    assert [item.index for item in result.items] == [0, 1]


@pytest.mark.parametrize(
    ("items", "item_count"),
    [
        ([{"index": 0, "bbox": [0.1, 0.2, 0.4, 0.5], "confidence": 0.9}], 2),
        (
            [
                {"index": 0, "bbox": [0.1, 0.2, 0.4, 0.5], "confidence": 0.9},
                {"index": 0, "bbox": [0.5, 0.2, 0.8, 0.5], "confidence": 0.9},
            ],
            2,
        ),
        ([{"index": 2, "bbox": [0.1, 0.2, 0.4, 0.5], "confidence": 0.9}], 1),
    ],
)
def test_rejects_missing_duplicate_and_out_of_range_indexes(items, item_count):
    from app.services.vision_recognition import (
        LocalizationResult,
        VisionRecognitionError,
        validated_localizations,
    )

    result = LocalizationResult(items=items)

    with pytest.raises(VisionRecognitionError):
        validated_localizations(result, item_count=item_count)
