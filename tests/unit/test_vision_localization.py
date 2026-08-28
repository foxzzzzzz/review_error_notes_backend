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
    from app.services.vision_recognition import (
        LOCALIZATION_PROMPT,
        RECOGNITION_PROMPT,
        recognition_prompt_for,
    )

    assert "印刷提示、学生答案和相关红色批改标记" in LOCALIZATION_PROMPT
    assert "未标记的相邻兄弟小题" in LOCALIZATION_PROMPT
    assert "每个 index 恰好返回一次" in LOCALIZATION_PROMPT
    assert "matched=false" in LOCALIZATION_PROMPT
    assert "在整张图片中独立定位" in LOCALIZATION_PROMPT
    assert "recognition_bbox" not in LOCALIZATION_PROMPT
    assert "保持重叠" not in LOCALIZATION_PROMPT
    assert "tags 只能使用中文标签" in RECOGNITION_PROMPT
    marked_prompt = recognition_prompt_for("marked", "chinese", [])
    assert "一次判错事件" in marked_prompt
    assert "cross_bbox" in marked_prompt
    assert "circle_bbox" in marked_prompt
    assert "不得分别输出" in marked_prompt


def test_error_mark_accepts_cross_circle_components():
    from app.services.vision_recognition import ErrorMark

    mark = ErrorMark(
        mark_id=0,
        mark_type="cross_circle",
        bbox=[0.1, 0.1, 0.4, 0.4],
        cross_bbox=[0.3, 0.1, 0.4, 0.2],
        circle_bbox=[0.1, 0.2, 0.35, 0.4],
        confidence=0.95,
    )

    assert mark.cross_bbox == [0.3, 0.1, 0.4, 0.2]
    assert mark.circle_bbox == [0.1, 0.2, 0.35, 0.4]


def test_mark_question_localization_requires_bbox_only_when_matched():
    from app.services.vision_recognition import MarkQuestionLocalizationItem

    matched = MarkQuestionLocalizationItem(
        mark_id=2,
        matched=True,
        bbox=[0.1, 0.2, 0.4, 0.5],
        confidence=0.9,
    )
    unmatched = MarkQuestionLocalizationItem(
        mark_id=3,
        matched=False,
        bbox=None,
        confidence=0.4,
    )

    assert matched.bbox == [0.1, 0.2, 0.4, 0.5]
    assert unmatched.bbox is None


def test_mark_question_localization_rejects_duplicate_mark_ids():
    from app.services.vision_recognition import (
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
    )

    item = MarkQuestionLocalizationItem(
        mark_id=2,
        matched=True,
        bbox=[0.1, 0.2, 0.4, 0.5],
        confidence=0.9,
    )
    with pytest.raises(Exception):
        MarkQuestionLocalizationResult(items=[item, item])


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


def test_localize_retries_invalid_json_with_format_correction(tmp_path, caplog):
    from app.services.vision_recognition import MiniMaxVisionClient

    source = tmp_path / "question.jpg"
    _write_image(source)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        content = "private-invalid-localization-json"
        if len(requests) == 2:
            content = json.dumps(
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
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return httpx.Response(
            200,
            json={"content": content, "base_resp": {"status_code": 0}},
        )

    caplog.set_level("INFO")
    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=1,
        max_edge=1200,
        jpeg_quality=90,
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    result = client.localize(
        str(source),
        [_vision_item("kè wén", "课文", "write_pinyin")],
        [_error_mark()],
    )

    assert result.items[0].index == 0
    assert len(requests) == 2
    assert "格式纠偏" in requests[1]["prompt"]
    assert (
        "vision_response_retry operation=localization "
        "error_code=vision_response_json_invalid attempt=1 max_retries=1"
    ) in caplog.text
    assert "private-invalid-localization-json" not in caplog.text


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

    with pytest.raises(VisionRecognitionError) as exc_info:
        validated_localizations(result, item_count=item_count, marks={0: _error_mark()})

    assert exc_info.value.code == "vision_localization_invalid"
    assert exc_info.value.diagnostic["reason"] in {
        "index_count_mismatch",
        "index_set_mismatch",
    }
    assert exc_info.value.diagnostic["item_count"] == item_count


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


def test_geometry_diagnostic_distinguishes_area_and_mark_center_failures():
    from app.services.vision_recognition import (
        LocalizationItem,
        localization_geometry_diagnostic,
    )

    oversized = LocalizationItem(
        index=0,
        matched=True,
        mark_ids=[0],
        bbox=[0.0, 0.0, 1.0, 0.8],
        observed_prompt_text="课文",
        observed_raw_text="kè wén",
        confidence=0.96,
    )
    outside_mark = LocalizationItem(
        index=1,
        matched=True,
        mark_ids=[0],
        bbox=[0.6, 0.6, 0.8, 0.8],
        observed_prompt_text="课文",
        observed_raw_text="kè wén",
        confidence=0.96,
    )

    oversized_diagnostic = localization_geometry_diagnostic(
        oversized,
        marks={0: _error_mark()},
        max_area_ratio=0.35,
    )
    outside_diagnostic = localization_geometry_diagnostic(
        outside_mark,
        marks={0: _error_mark()},
        max_area_ratio=0.35,
    )

    assert oversized_diagnostic == {
        "passed": False,
        "bbox_area_ratio": 0.8,
        "max_area_ratio": 0.35,
        "mark_ids": [0],
        "missing_mark_ids": [],
        "outside_mark_ids": [],
        "outside_mark_diagnostics": [],
        "failure_reasons": ["bbox_area_exceeded"],
    }
    assert outside_diagnostic == {
        "passed": False,
        "bbox_area_ratio": 0.04,
        "max_area_ratio": 0.35,
        "mark_ids": [0],
        "missing_mark_ids": [],
        "outside_mark_ids": [0],
        "outside_mark_diagnostics": [
            {
                "mark_id": 0,
                "horizontal_gap_ratio": 0.35,
                "vertical_gap_ratio": 0.3,
                "nearest_distance_ratio": 0.460977,
                "mark_bbox_intersects_question_bbox": False,
            }
        ],
        "failure_reasons": ["mark_center_outside_bbox"],
    }


def test_cross_circle_uses_circle_as_question_anchor():
    from app.services.vision_recognition import (
        ErrorMark,
        LocalizationItem,
        localization_geometry_diagnostic,
    )

    localization = LocalizationItem(
        index=0,
        matched=True,
        mark_ids=[0],
        bbox=[0.2, 0.3, 0.5, 0.5],
        observed_prompt_text="词语",
        observed_raw_text="作答",
        confidence=0.95,
    )
    mark = ErrorMark(
        mark_id=0,
        mark_type="cross_circle",
        bbox=[0.22, 0.2, 0.48, 0.48],
        cross_bbox=[0.4, 0.2, 0.48, 0.28],
        circle_bbox=[0.22, 0.3, 0.46, 0.48],
        confidence=0.95,
    )

    diagnostic = localization_geometry_diagnostic(
        localization,
        marks={0: mark},
        max_area_ratio=0.35,
        anchor_max_gap_ratio=0.08,
        cross_only_max_gap_ratio=0.08,
    )

    assert diagnostic["passed"] is True
    assert diagnostic["anchor_diagnostics"][0]["anchor_type"] == "circle"
    assert diagnostic["anchor_diagnostics"][0]["intersects_question_bbox"] is True


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([0.54, 0.3, 0.58, 0.36], True),
        ([0.68, 0.3, 0.72, 0.36], False),
    ],
)
def test_cross_only_uses_bounded_distance(bbox, expected):
    from app.services.vision_recognition import (
        ErrorMark,
        LocalizationItem,
        localization_geometry_diagnostic,
    )

    localization = LocalizationItem(
        index=0,
        matched=True,
        mark_ids=[0],
        bbox=[0.2, 0.3, 0.5, 0.5],
        observed_prompt_text="词语",
        observed_raw_text="作答",
        confidence=0.95,
    )
    mark = ErrorMark(
        mark_id=0,
        mark_type="cross",
        bbox=bbox,
        confidence=0.95,
    )

    diagnostic = localization_geometry_diagnostic(
        localization,
        marks={0: mark},
        max_area_ratio=0.35,
        anchor_max_gap_ratio=0.08,
        cross_only_max_gap_ratio=0.08,
    )

    assert diagnostic["passed"] is expected


def test_repairs_unique_missing_mark_assignment_with_circle_anchor():
    from app.services.vision_recognition import (
        ErrorMark,
        LocalizationItem,
        repair_unique_mark_assignments,
    )

    items = [
        _vision_item("甲", "第一题", "write_word"),
        _vision_item("乙", "第二题", "write_word"),
    ]
    localizations = {
        0: LocalizationItem(
            index=0,
            matched=True,
            mark_ids=[],
            bbox=[0.05, 0.1, 0.35, 0.3],
            observed_prompt_text="第一题",
            observed_raw_text="甲",
            confidence=0.95,
        ),
        1: LocalizationItem(
            index=1,
            matched=True,
            mark_ids=[],
            bbox=[0.55, 0.1, 0.85, 0.3],
            observed_prompt_text="第二题",
            observed_raw_text="乙",
            confidence=0.95,
        ),
    }
    mark = ErrorMark(
        mark_id=0,
        mark_type="cross_circle",
        bbox=[0.6, 0.05, 0.82, 0.28],
        cross_bbox=[0.75, 0.05, 0.82, 0.12],
        circle_bbox=[0.6, 0.12, 0.8, 0.28],
        confidence=0.96,
    )

    repaired, diagnostics = repair_unique_mark_assignments(
        localizations,
        {0: mark},
        items,
        localization_threshold=0.8,
        max_area_ratio=0.35,
        anchor_max_gap_ratio=0.08,
        cross_only_max_gap_ratio=0.08,
    )

    assert repaired[0].mark_ids == []
    assert repaired[1].mark_ids == [0]
    assert diagnostics == [
        {
            "index": 1,
            "mark_id": 0,
            "assignment_source": "deterministic",
            "anchor_type": "circle",
            "nearest_distance_ratio": 0.0,
        }
    ]


def test_does_not_repair_ambiguous_or_already_assigned_marks():
    from app.services.vision_recognition import (
        LocalizationItem,
        repair_unique_mark_assignments,
    )

    items = [
        _vision_item("甲", "第一题", "write_word"),
        _vision_item("乙", "第二题", "write_word"),
    ]
    ambiguous = {
        0: LocalizationItem(
            index=0,
            matched=True,
            mark_ids=[],
            bbox=[0.1, 0.1, 0.4, 0.3],
            observed_prompt_text="第一题",
            observed_raw_text="甲",
            confidence=0.95,
        ),
        1: LocalizationItem(
            index=1,
            matched=True,
            mark_ids=[],
            bbox=[0.6, 0.1, 0.9, 0.3],
            observed_prompt_text="第二题",
            observed_raw_text="乙",
            confidence=0.95,
        ),
    }
    centered_mark = _error_mark(0, [0.48, 0.15, 0.52, 0.25])

    repaired, diagnostics = repair_unique_mark_assignments(
        ambiguous,
        {0: centered_mark},
        items,
        localization_threshold=0.8,
        max_area_ratio=0.35,
        anchor_max_gap_ratio=0.08,
        cross_only_max_gap_ratio=0.08,
    )
    assert all(not localization.mark_ids for localization in repaired.values())
    assert diagnostics == []

    occupied = dict(ambiguous)
    occupied[0] = occupied[0].model_copy(update={"mark_ids": [0]})
    repaired, diagnostics = repair_unique_mark_assignments(
        occupied,
        {0: centered_mark},
        items,
        localization_threshold=0.8,
        max_area_ratio=0.35,
        anchor_max_gap_ratio=0.08,
        cross_only_max_gap_ratio=0.08,
    )
    assert repaired[1].mark_ids == []
    assert diagnostics == []


def test_does_not_repair_far_missing_mark():
    from app.services.vision_recognition import (
        LocalizationItem,
        repair_unique_mark_assignments,
    )

    item = _vision_item("甲", "第一题", "write_word")
    localization = LocalizationItem(
        index=0,
        matched=True,
        mark_ids=[],
        bbox=[0.1, 0.1, 0.3, 0.3],
        observed_prompt_text="第一题",
        observed_raw_text="甲",
        confidence=0.95,
    )

    repaired, diagnostics = repair_unique_mark_assignments(
        {0: localization},
        {0: _error_mark(0, [0.7, 0.7, 0.8, 0.8])},
        [item],
        localization_threshold=0.8,
        max_area_ratio=0.35,
        anchor_max_gap_ratio=0.08,
        cross_only_max_gap_ratio=0.08,
    )
    assert repaired[0].mark_ids == []
    assert diagnostics == []


def test_geometry_diagnostic_reports_missing_and_unknown_mark_ids():
    from app.services.vision_recognition import (
        LocalizationItem,
        localization_geometry_diagnostic,
    )

    missing_assignment = LocalizationItem(
        index=0,
        matched=True,
        mark_ids=[],
        bbox=[0.1, 0.1, 0.4, 0.4],
        observed_prompt_text="课文",
        observed_raw_text="kè wén",
        confidence=0.96,
    )
    unknown_assignment = missing_assignment.model_copy(
        update={"index": 1, "mark_ids": [7]}
    )

    assert localization_geometry_diagnostic(
        missing_assignment,
        marks={0: _error_mark()},
        max_area_ratio=0.35,
    )["failure_reasons"] == ["missing_mark_ids"]
    unknown_diagnostic = localization_geometry_diagnostic(
        unknown_assignment,
        marks={0: _error_mark()},
        max_area_ratio=0.35,
    )
    assert unknown_diagnostic["missing_mark_ids"] == [7]
    assert unknown_diagnostic["failure_reasons"] == ["unknown_mark_ids"]


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

    with pytest.raises(VisionRecognitionError) as exc_info:
        validated_localizations(result, item_count=2, marks={0: _error_mark()})

    assert exc_info.value.code == "vision_localization_invalid"
    assert exc_info.value.diagnostic["reason"] == "duplicate_mark_assignment"


def test_allows_valid_mark_that_was_not_assigned_to_any_item():
    from app.services.vision_recognition import (
        LocalizationResult,
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

    localizations = validated_localizations(
        result,
        item_count=1,
        marks={
            0: _error_mark(),
            1: _error_mark(1, [0.6, 0.25, 0.7, 0.35]),
        },
    )

    assert list(localizations) == [0]


def test_rejects_mark_assignment_that_is_not_in_valid_marks():
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
                "mark_ids": [7],
                "bbox": [0.1, 0.2, 0.4, 0.5],
                "observed_prompt_text": "课文",
                "observed_raw_text": "kè wén",
                "confidence": 0.9,
            }
        ]
    )

    with pytest.raises(VisionRecognitionError) as exc_info:
        validated_localizations(result, item_count=1, marks={0: _error_mark()})

    assert exc_info.value.code == "vision_localization_invalid"
    assert exc_info.value.diagnostic["reason"] == "unknown_mark_assignment"


def test_local_red_rescue_rejects_small_noise_component():
    from app.services.error_mark_validation import RedMarkRegion
    from app.services.vision_recognition import (
        LocalizationItem,
        repair_unique_local_red_assignments,
    )

    localizations = {
        0: LocalizationItem(
            index=0,
            matched=True,
            mark_ids=[],
            bbox=[0.2, 0.2, 0.5, 0.5],
            observed_prompt_text="课文",
            observed_raw_text="kè wén",
            confidence=0.95,
        )
    }
    region = RedMarkRegion(
        bbox=[0.25, 0.25, 0.3, 0.3],
        pixel_count=20,
        area_ratio=0.0025,
        thinness_ratio=1.0,
    )

    rescued, diagnostics = repair_unique_local_red_assignments(
        localizations,
        {},
        [region],
        localization_threshold=0.85,
        max_area_ratio=0.35,
        max_gap_ratio=0.08,
        min_pixels=80,
    )

    assert rescued == set()
    assert diagnostics == []
