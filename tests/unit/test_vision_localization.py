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
    )


def _error_mark(mark_id=0, bbox=None):
    from app.services.vision_recognition import ErrorMark

    return ErrorMark(
        mark_id=mark_id,
        mark_type="circle",
        bbox=bbox or [0.2, 0.25, 0.3, 0.35],
        confidence=0.96,
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
    assert "matched=false" in LOCALIZATION_PROMPT
    assert "在整张图片中独立定位" in LOCALIZATION_PROMPT
    assert "recognition_bbox" not in LOCALIZATION_PROMPT
    assert "保持重叠" not in LOCALIZATION_PROMPT
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
                            {
                                "index": 0,
                                "matched": True,
                                "mark_ids": [0],
                                "bbox": [0.1, 0.2, 0.4, 0.5],
                                "observed_prompt_text": "课文",
                                "observed_raw_text": "kè wén",
                                "confidence": 0.94,
                            },
                            {
                                "index": 1,
                                "matched": True,
                                "mark_ids": [1],
                                "bbox": [0.5, 0.2, 0.8, 0.5],
                                "observed_prompt_text": "hé zuò",
                                "observed_raw_text": "合作",
                                "confidence": 0.91,
                            },
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
    marks = [_error_mark(), _error_mark(1, [0.6, 0.25, 0.7, 0.35])]

    result = _client(handler).localize(str(source), items, marks)

    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["image_url"].startswith("data:image/jpeg;base64,")
    assert '"index": 0' in body["prompt"]
    assert '"index": 1' in body["prompt"]
    assert '"mark_id": 0' in body["prompt"]
    assert '"prompt_text": "课文"' in body["prompt"]
    assert "recognition_bbox" not in body["prompt"]
    assert [item.index for item in result.items] == [0, 1]


def test_localization_allows_an_explicit_unmatched_result_without_bbox():
    from app.services.vision_recognition import LocalizationItem

    item = LocalizationItem(
        index=0,
        matched=False,
        mark_ids=[],
        bbox=None,
        observed_prompt_text=None,
        observed_raw_text=None,
        confidence=0.96,
    )

    assert item.matched is False
    assert item.bbox is None


@pytest.mark.parametrize(
    ("items", "item_count"),
    [
        (
            [
                {
                    "index": 0,
                    "matched": True,
                    "mark_ids": [0],
                    "bbox": [0.1, 0.2, 0.4, 0.5],
                    "observed_prompt_text": "课文",
                    "observed_raw_text": "kè wén",
                    "confidence": 0.9,
                }
            ],
            2,
        ),
        (
            [
                {
                    "index": 0,
                    "matched": True,
                    "mark_ids": [0],
                    "bbox": [0.1, 0.2, 0.4, 0.5],
                    "observed_prompt_text": "课文",
                    "observed_raw_text": "kè wén",
                    "confidence": 0.9,
                },
                {
                    "index": 0,
                    "matched": True,
                    "mark_ids": [1],
                    "bbox": [0.5, 0.2, 0.8, 0.5],
                    "observed_prompt_text": "合作",
                    "observed_raw_text": "hé zuò",
                    "confidence": 0.9,
                },
            ],
            2,
        ),
        (
            [
                {
                    "index": 2,
                    "matched": True,
                    "mark_ids": [0],
                    "bbox": [0.1, 0.2, 0.4, 0.5],
                    "observed_prompt_text": "课文",
                    "observed_raw_text": "kè wén",
                    "confidence": 0.9,
                }
            ],
            1,
        ),
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
        validated_localizations(result, item_count=item_count, marks={0: _error_mark()})


def test_localization_requires_assigned_mark_center_inside_bbox():
    from app.services.vision_recognition import (
        LocalizationItem,
        localization_passes_geometry,
    )

    localization = LocalizationItem(
        index=0,
        matched=True,
        mark_ids=[0],
        bbox=[0.6, 0.6, 0.8, 0.8],
        observed_prompt_text="课文",
        observed_raw_text="kè wén",
        confidence=0.96,
    )

    assert not localization_passes_geometry(
        localization,
        marks={0: _error_mark()},
        max_area_ratio=0.35,
    )


def test_localization_rejects_oversized_bbox():
    from app.services.vision_recognition import (
        LocalizationItem,
        localization_passes_geometry,
    )

    localization = LocalizationItem(
        index=0,
        matched=True,
        mark_ids=[0],
        bbox=[0.0, 0.0, 1.0, 0.8],
        observed_prompt_text="课文",
        observed_raw_text="kè wén",
        confidence=0.96,
    )

    assert not localization_passes_geometry(
        localization,
        marks={0: _error_mark()},
        max_area_ratio=0.35,
    )


def test_rejects_duplicate_mark_assignment_across_items():
    from app.services.vision_recognition import (
        LocalizationResult,
        VisionRecognitionError,
        validated_localizations,
    )

    result = LocalizationResult(
        items=[
            {
                "index": index,
                "matched": True,
                "mark_ids": [0],
                "bbox": [0.1 + index * 0.3, 0.2, 0.35 + index * 0.3, 0.5],
                "observed_prompt_text": "课文",
                "observed_raw_text": "kè wén",
                "confidence": 0.9,
            }
            for index in range(2)
        ]
    )

    with pytest.raises(VisionRecognitionError):
        validated_localizations(result, item_count=2, marks={0: _error_mark()})


def test_rejects_valid_mark_that_was_not_assigned_to_any_item():
    from app.services.vision_recognition import (
        LocalizationResult,
        VisionRecognitionError,
        validated_localizations,
    )

    result = LocalizationResult(
        items=[
            {
                "index": 0,
                "matched": True,
                "mark_ids": [0],
                "bbox": [0.1, 0.2, 0.4, 0.5],
                "observed_prompt_text": "课文",
                "observed_raw_text": "kè wén",
                "confidence": 0.9,
            }
        ]
    )

    with pytest.raises(VisionRecognitionError):
        validated_localizations(
            result,
            item_count=1,
            marks={
                0: _error_mark(),
                1: _error_mark(1, [0.6, 0.25, 0.7, 0.35]),
            },
        )
