import pytest
from PIL import Image, ImageDraw

from app.services.local_ocr_verification import OCRVerification
from app.services.vision_recognition import (
    ErrorMark,
    LocalizationItem,
    LocalizationResult,
    VisionItem,
    VisionRecognitionError,
    VisionResult,
)


def _write_tag_config(tmp_path):
    config_path = tmp_path / "tag-aliases.json"
    config_path.write_text(
        """
{
  "aliases": {
    "pinyin": "拼音",
    "teacher-marked": "老师批改",
    "word": "词语",
    "wrong-character": "错别字"
  },
  "question_type_defaults": {
    "write_pinyin": "拼音",
    "write_word": "词语"
  }
}
""".strip(),
        encoding="utf-8",
    )
    return str(config_path)


def _write_source_image(tmp_path):
    image_path = tmp_path / "question.jpg"
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 70, 120, 100), fill=(220, 30, 30))
    draw.rectangle((250, 180, 290, 215), fill=(220, 30, 30))
    image.save(image_path)
    return str(image_path)


def _vision_result():
    return VisionResult(
        items=[
            VisionItem(
                raw_text="kè wén",
                instruction="看词语写拼音",
                prompt_text="课文",
                normalized_text="kè wén",
                answer="kè wén",
                subject="chinese",
                question_type="write_pinyin",
                tags=["pinyin", "teacher-marked"],
                difficulty=2,
                confidence=0.95,
                uncertain_segments=[],
            ),
            VisionItem(
                raw_text="合做",
                instruction="看拼音写词语",
                prompt_text="hé zuò",
                normalized_text="合作",
                answer="合作",
                subject="chinese",
                question_type="write_word",
                tags=["word", "wrong-character"],
                difficulty=2,
                confidence=0.92,
                uncertain_segments=[],
            ),
        ],
        error_marks=[
            ErrorMark(
                mark_id=0,
                mark_type="circle",
                bbox=[0.2, 0.23, 0.3, 0.34],
                confidence=0.96,
            ),
            ErrorMark(
                mark_id=1,
                mark_type="cross",
                bbox=[0.62, 0.6, 0.73, 0.72],
                confidence=0.95,
            ),
        ],
        ignored_text=[],
    )


class FakeClient:
    def __init__(self, localization_error=False):
        self.recognize_calls = 0
        self.localize_calls = 0
        self.localization_error = localization_error
        self.localized_marks = None

    def recognize(self, image_path, subject_hint=None):
        self.recognize_calls += 1
        return _vision_result()

    def localize(self, image_path, items, error_marks):
        self.localize_calls += 1
        self.localized_marks = error_marks
        if self.localization_error:
            raise VisionRecognitionError("localization failed")
        return LocalizationResult(
            items=[
                LocalizationItem(
                    index=0,
                    matched=True,
                    mark_ids=[0],
                    bbox=[0.15, 0.15, 0.4, 0.45],
                    observed_prompt_text="课文",
                    observed_raw_text="kè wén",
                    confidence=0.94,
                ),
                LocalizationItem(
                    index=1,
                    matched=True,
                    mark_ids=[1],
                    bbox=[0.55, 0.5, 0.8, 0.8],
                    observed_prompt_text="hé zuò",
                    observed_raw_text="合做",
                    confidence=0.91,
                ),
            ]
        )


class FakeOCRVerifier:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def verify(self, image_path, bbox, target_index, items):
        self.calls.append((bbox, target_index))
        return self.results.get(
            target_index,
            OCRVerification(status="inconclusive"),
        )


def _run_batch(
    tmp_path,
    client=None,
    ocr_verifier=None,
    crop_context_padding_ratio=0.0,
    local_red_scan=None,
    mark_mismatch_retry_count=0,
    force_mode=None,
    correction_group_enabled=False,
    pair_max_distance_ratio=0.12,
    dedup_iou_threshold=0.8,
    anchor_max_gap_ratio=0.08,
    cross_only_max_gap_ratio=0.08,
    semantic_retry_count=0,
    marked_ocr_recheck_limit=0,
    local_red_rescue_min_pixels=80,
    three_stage_enabled=False,
):
    from app.services.vision_recognition import recognize_question_batch

    return recognize_question_batch(
        client=client or FakeClient(),
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        confidence_threshold=0.85,
        mark_confidence_threshold=0.85,
        localization_threshold=0.85,
        localization_max_area_ratio=0.35,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        tag_config_path=_write_tag_config(tmp_path),
        ocr_verifier=ocr_verifier or FakeOCRVerifier(),
        crop_context_padding_ratio=crop_context_padding_ratio,
        local_red_scan=local_red_scan,
        mark_mismatch_retry_count=mark_mismatch_retry_count,
        force_mode=force_mode,
        correction_group_enabled=correction_group_enabled,
        pair_max_distance_ratio=pair_max_distance_ratio,
        dedup_iou_threshold=dedup_iou_threshold,
        anchor_max_gap_ratio=anchor_max_gap_ratio,
        cross_only_max_gap_ratio=cross_only_max_gap_ratio,
        semantic_retry_count=semantic_retry_count,
        marked_ocr_recheck_limit=marked_ocr_recheck_limit,
        local_red_rescue_min_pixels=local_red_rescue_min_pixels,
        three_stage_enabled=three_stage_enabled,
    )


def _detected_red_scan():
    from app.services.error_mark_validation import RedMarkRegion, RedMarkScanResult

    return RedMarkScanResult(
        status="detected",
        regions=[
            RedMarkRegion(
                bbox=[0.2, 0.23, 0.3, 0.34],
                pixel_count=100,
                area_ratio=0.011,
                thinness_ratio=1.1,
            )
        ],
        red_pixel_count=100,
        scanned_width=400,
        scanned_height=300,
        duration_ms=1.0,
    )


def test_three_stage_marked_recognition_keeps_stable_mark_ids(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
        recognize_marked_three_stage,
    )

    class ThreeStageClient:
        def __init__(self):
            self.calls = []

        def detect_marks(self, image_path, local_red_regions, correction=None):
            self.calls.append("marks")
            return MarkDetectionResult(error_marks=_vision_result().error_marks)

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            self.calls.append("localization")
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=mark.mark_id,
                        matched=True,
                        bbox=(
                            [0.15, 0.15, 0.4, 0.45]
                            if mark.mark_id == 0
                            else [0.55, 0.5, 0.8, 0.8]
                        ),
                        confidence=0.94,
                    )
                    for mark in error_marks
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            self.calls.append("content")
            assert mark_ids == [0, 1]
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=mark_id,
                        **_vision_result().items[mark_id].model_dump(),
                    )
                    for mark_id in mark_ids
                ]
            )

    client = ThreeStageClient()
    result, localizations, marks, diagnostic = recognize_marked_three_stage(
        client=client,
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        local_red_regions=[],
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        pair_max_distance_ratio=0.04,
        dedup_iou_threshold=0.8,
        crop_context_padding_ratio=0.1,
        image_max_edge=1200,
        image_jpeg_quality=90,
        image_max_pixels=40_000_000,
    )

    assert client.calls == ["marks", "localization", "content"]
    assert len(result.items) == 2
    assert [mark.mark_id for mark in marks] == [0, 1]
    assert localizations[0].mark_ids == [0]
    assert localizations[1].mark_ids == [1]
    assert diagnostic["content_item_count"] == 2


def test_three_stage_keeps_successful_items_when_one_mark_is_unmatched(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
        recognize_marked_three_stage,
    )

    class PartialClient:
        def detect_marks(self, image_path, local_red_regions, correction=None):
            return MarkDetectionResult(error_marks=_vision_result().error_marks)

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=0,
                        matched=True,
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        confidence=0.94,
                    ),
                    MarkQuestionLocalizationItem(
                        mark_id=1,
                        matched=False,
                        bbox=None,
                        confidence=0.4,
                    ),
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=0,
                        **_vision_result().items[0].model_dump(),
                    )
                ]
            )

    result, localizations, _marks, diagnostic = recognize_marked_three_stage(
        client=PartialClient(),
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        local_red_regions=[],
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        pair_max_distance_ratio=0.04,
        dedup_iou_threshold=0.8,
        crop_context_padding_ratio=0.1,
        image_max_edge=1200,
        image_jpeg_quality=90,
        image_max_pixels=40_000_000,
    )

    assert len(result.items) == 1
    assert set(localizations) == {0}
    assert diagnostic["unlocalized_mark_ids"] == []
    assert diagnostic["missing_content_mark_ids"] == [1]


def test_three_stage_uses_circle_context_and_preserves_complete_question_bbox(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
        recognize_marked_three_stage,
    )

    class CircleContextClient:
        def __init__(self):
            self.context_calls = 0

        def detect_marks(self, image_path, local_red_regions, correction=None):
            return MarkDetectionResult(
                error_marks=[
                    ErrorMark(
                        mark_id=0,
                        mark_type="circle",
                        bbox=[0.2, 0.23, 0.3, 0.34],
                        confidence=0.96,
                    ),
                    ErrorMark(
                        mark_id=1,
                        mark_type="cross",
                        bbox=[0.22, 0.22, 0.28, 0.3],
                        confidence=0.95,
                    ),
                ]
            )

        def locate_marked_question_context(self, image_path, error_mark, correction=None):
            self.context_calls += 1
            assert error_mark.mark_type == "cross_circle"
            assert error_mark.circle_bbox[0] == pytest.approx(1 / 3)
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=error_mark.mark_id,
                        matched=True,
                        answer_bbox=[0.3, 0.3, 0.7, 0.7],
                        prompt_bbox=[0.2, 0.05, 0.8, 0.25],
                        question_bbox=[0.1, 0.04, 0.9, 0.9],
                        confidence=0.94,
                    )
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=0,
                        **_vision_result().items[0].model_dump(),
                    )
                ]
            )

    client = CircleContextClient()
    result, localizations, marks, diagnostic = recognize_marked_three_stage(
        client=client,
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        local_red_regions=[],
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        pair_max_distance_ratio=0.04,
        dedup_iou_threshold=0.8,
        crop_context_padding_ratio=0.1,
        image_max_edge=1200,
        image_jpeg_quality=90,
        image_max_pixels=40_000_000,
    )

    assert len(result.items) == 1
    assert [mark.mark_type for mark in marks] == ["cross_circle"]
    assert client.context_calls == 1
    assert localizations[0].answer_bbox == pytest.approx(
        [0.19, 0.22, 0.31, 0.3533333333]
    )
    assert localizations[0].bbox == pytest.approx(
        [0.13, 0.1333333333, 0.37, 0.42]
    )
    assert localizations[0].observed_prompt_text is None
    assert localizations[0].observed_raw_text is None
    assert diagnostic["per_mark_localization"][0]["bbox_source"] == "circle_tolerant_llm"


def test_nearby_circle_cross_pair_is_forced_to_pending_review(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
        recognize_marked_three_stage,
    )

    class NearbyPairClient:
        def detect_marks(self, image_path, local_red_regions, correction=None):
            return MarkDetectionResult(
                error_marks=[
                    ErrorMark(
                        mark_id=0,
                        mark_type="circle",
                        bbox=[0.20, 0.23, 0.25, 0.34],
                        confidence=0.96,
                    ),
                    ErrorMark(
                        mark_id=1,
                        mark_type="cross",
                        bbox=[0.26, 0.23, 0.30, 0.34],
                        confidence=0.95,
                    ),
                ]
            )

        def locate_marked_question_context(self, image_path, error_mark, correction=None):
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=error_mark.mark_id,
                        matched=True,
                        answer_bbox=[0.30, 0.30, 0.70, 0.70],
                        question_bbox=[0.10, 0.10, 0.90, 0.90],
                        confidence=0.94,
                    )
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=0,
                        **_vision_result().items[0].model_dump(),
                    )
                ]
            )

    _result, localizations, marks, diagnostic = recognize_marked_three_stage(
        client=NearbyPairClient(),
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        local_red_regions=[],
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        pair_max_distance_ratio=0.04,
        dedup_iou_threshold=0.8,
        crop_context_padding_ratio=0.1,
        image_max_edge=1200,
        image_jpeg_quality=90,
        image_max_pixels=40_000_000,
    )

    assert [mark.mark_type for mark in marks] == ["cross_circle"]
    assert diagnostic["mark_grouping"]["review_required_mark_ids"] == [0]
    assert localizations[0].localization_status == "needs_review"


def test_three_stage_uses_targeted_region_detection_before_pairing(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
        RegionalErrorMark,
        RegionalMarkDetectionResult,
        recognize_marked_three_stage,
    )

    class TargetedRegionClient:
        def __init__(self):
            self.targeted_calls = 0

        def detect_marks(self, image_path, local_red_regions, correction=None):
            return MarkDetectionResult(
                error_marks=[
                    ErrorMark(
                        mark_id=0,
                        mark_type="cross",
                        bbox=[0.20, 0.23, 0.30, 0.34],
                        confidence=0.9,
                    )
                ]
            )

        def detect_marks_in_regions(self, image_path, region_ids, correction=None):
            self.targeted_calls += 1
            assert region_ids == [0]
            return RegionalMarkDetectionResult(
                error_marks=[
                    RegionalErrorMark(
                        region_id=0,
                        mark_id=0,
                        mark_type="circle",
                        bbox=[0.27, 0.10, 0.73, 0.90],
                        confidence=0.96,
                    ),
                    RegionalErrorMark(
                        region_id=0,
                        mark_id=1,
                        mark_type="cross",
                        bbox=[0.45, 0.20, 0.65, 0.80],
                        confidence=0.95,
                    ),
                ]
            )

        def locate_marked_question_context(self, image_path, error_mark, correction=None):
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=error_mark.mark_id,
                        matched=True,
                        answer_bbox=[0.30, 0.30, 0.70, 0.70],
                        question_bbox=[0.10, 0.10, 0.90, 0.90],
                        confidence=0.94,
                    )
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=mark_ids[0],
                        **_vision_result().items[0].model_dump(),
                    )
                ]
            )

    client = TargetedRegionClient()
    _result, _localizations, marks, diagnostic = recognize_marked_three_stage(
        client=client,
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        local_red_regions=[[0.20, 0.23, 0.30, 0.34]],
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        pair_max_distance_ratio=0.04,
        dedup_iou_threshold=0.8,
        crop_context_padding_ratio=0.1,
        image_max_edge=1200,
        image_jpeg_quality=90,
        image_max_pixels=40_000_000,
        mark_stage_retry_count=1,
        evidence_context_min_width_ratio=0.22,
        evidence_context_min_height_ratio=0.14,
    )

    assert client.targeted_calls == 1
    assert [mark.mark_type for mark in marks] == ["cross_circle"]
    assert diagnostic["targeted_mark_llm_attempts"] == 1
    assert diagnostic["targeted_region_count"] == 1


def test_three_stage_rescues_uncovered_pixel_rich_region_for_full_page_localization(
    tmp_path,
):
    from app.services.error_mark_validation import RedMarkRegion
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
        recognize_marked_three_stage,
    )

    class UncoveredRegionClient:
        def detect_marks(self, image_path, local_red_regions, correction=None):
            return MarkDetectionResult(error_marks=[])

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            assert [mark.mark_type for mark in error_marks] == ["mixed"]
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=error_marks[0].mark_id,
                        matched=True,
                        bbox=[0.15, 0.12, 0.43, 0.38],
                        confidence=0.9,
                    )
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=mark_ids[0],
                        **_vision_result().items[0].model_dump(),
                    )
                ]
            )

    _result, localizations, marks, diagnostic = recognize_marked_three_stage(
        client=UncoveredRegionClient(),
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        local_red_regions=[[0.20, 0.23, 0.30, 0.34]],
        local_red_evidence_regions=[
            RedMarkRegion(
                bbox=[0.20, 0.23, 0.30, 0.34],
                pixel_count=120,
                area_ratio=0.011,
                thinness_ratio=1.1,
            )
        ],
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        pair_max_distance_ratio=0.04,
        dedup_iou_threshold=0.8,
        crop_context_padding_ratio=0.1,
        image_max_edge=1200,
        image_jpeg_quality=90,
        image_max_pixels=40_000_000,
        mark_stage_retry_count=0,
        local_red_rescue_min_pixels=80,
    )

    assert [mark.mark_type for mark in marks] == ["mixed"]
    assert localizations[0].localization_status == "needs_review"
    assert diagnostic["rescued_uncovered_region_ids"] == [0]
    assert diagnostic["uncovered_local_red_region_count"] == 0
    assert diagnostic["localized_question_geometry"][0]["bbox"] == [
        0.15,
        0.12,
        0.43,
        0.38,
    ]


def test_circle_context_retries_only_the_edge_clipped_mark(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
        recognize_marked_three_stage,
    )

    class RetryContextClient:
        def __init__(self):
            self.context_calls = 0

        def detect_marks(self, image_path, local_red_regions, correction=None):
            return MarkDetectionResult(
                error_marks=[
                    ErrorMark(
                        mark_id=0,
                        mark_type="circle",
                        bbox=[0.2, 0.23, 0.3, 0.34],
                        confidence=0.96,
                    )
                ]
            )

        def locate_marked_question_context(self, image_path, error_mark, correction=None):
            self.context_calls += 1
            question_bbox = (
                [0.0, 0.05, 0.9, 0.9]
                if self.context_calls == 1
                else [0.2, 0.2, 0.8, 0.8]
            )
            answer_bbox = (
                [0.35, 0.35, 0.65, 0.65]
                if self.context_calls == 1
                else [0.4, 0.4, 0.6, 0.6]
            )
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=error_mark.mark_id,
                        matched=True,
                        answer_bbox=answer_bbox,
                        prompt_bbox=None,
                        question_bbox=question_bbox,
                        confidence=0.94,
                    )
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=0,
                        **_vision_result().items[0].model_dump(),
                    )
                ]
            )

    client = RetryContextClient()
    _result, localizations, _marks, diagnostic = recognize_marked_three_stage(
        client=client,
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        local_red_regions=[],
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        pair_max_distance_ratio=0.04,
        dedup_iou_threshold=0.8,
        crop_context_padding_ratio=0.1,
        image_max_edge=1200,
        image_jpeg_quality=90,
        image_max_pixels=40_000_000,
        localization_stage_retry_count=1,
    )

    assert client.context_calls == 2
    assert localizations[0].bbox_source == "circle_tolerant_llm"
    assert diagnostic["per_mark_localization"][0]["attempts"][0][
        "failure_reasons"
    ] == ["question_bbox_touches_context_edge"]


def test_circle_context_retry_exhaustion_keeps_pending_fallback_candidate(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
    )

    class FallbackClient:
        def detect_marks(self, image_path, local_red_regions, correction=None):
            return MarkDetectionResult(
                error_marks=[
                    ErrorMark(
                        mark_id=0,
                        mark_type="circle",
                        bbox=[0.2, 0.23, 0.3, 0.34],
                        confidence=0.96,
                    )
                ]
            )

        def locate_marked_question_context(self, image_path, error_mark, correction=None):
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=error_mark.mark_id,
                        matched=False,
                        incomplete_reason="private student text must not be logged",
                        confidence=0.4,
                    )
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=0,
                        **_vision_result().items[0].model_dump(),
                    )
                ]
            )

    _result, values = _run_batch(
        tmp_path,
        client=FallbackClient(),
        local_red_scan=_detected_red_scan(),
        correction_group_enabled=True,
        three_stage_enabled=True,
    )

    assert len(values) == 1
    assert values[0]["collection_status"] == "pending_review"
    assert values[0]["review_status"] == "needs_review"
    assert values[0]["crop_region"]["bbox_source"] == "circle_tolerant_fallback"
    fallback_bbox = values[0]["crop_region"]["bbox"]
    assert fallback_bbox[2] - fallback_bbox[0] >= 0.22 - 1e-6
    assert fallback_bbox[3] - fallback_bbox[1] >= 0.14 - 1e-6
    assert values[0]["ocr_raw_json"]["three_stage"]["per_mark_localization"][0][
        "localization_status"
    ] == "needs_review"
    assert values[0]["ocr_raw_json"]["three_stage"]["per_mark_localization"][0][
        "attempts"
    ][0]["failure_reasons"] == ["incomplete_localization"]


def test_recognition_batch_uses_three_stage_path_without_legacy_calls(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
    )

    class ThreeStageOnlyClient:
        def __init__(self):
            self.calls = []

        def detect_marks(self, image_path, local_red_regions, correction=None):
            self.calls.append("marks")
            return MarkDetectionResult(error_marks=_vision_result().error_marks)

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            self.calls.append("localization")
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=0,
                        matched=True,
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        confidence=0.94,
                    ),
                    MarkQuestionLocalizationItem(
                        mark_id=1,
                        matched=True,
                        bbox=[0.55, 0.5, 0.8, 0.8],
                        confidence=0.94,
                    ),
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            self.calls.append("content")
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=mark_id,
                        **_vision_result().items[mark_id].model_dump(),
                    )
                    for mark_id in mark_ids
                ]
            )

    client = ThreeStageOnlyClient()
    result, values = _run_batch(
        tmp_path,
        client=client,
        local_red_scan=_detected_red_scan(),
        correction_group_enabled=True,
        three_stage_enabled=True,
    )

    assert client.calls == ["marks", "localization", "content"]
    assert len(result.items) == 2
    assert len(values) == 2
    assert all(value["ocr_raw_json"]["three_stage"]["content_item_count"] == 2 for value in values)


def test_three_stage_retries_only_missing_localization_and_content_ids(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
        recognize_marked_three_stage,
    )

    class RetryClient:
        def __init__(self):
            self.localization_calls = 0
            self.content_calls = 0

        def detect_marks(self, image_path, local_red_regions, correction=None):
            return MarkDetectionResult(error_marks=_vision_result().error_marks)

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            self.localization_calls += 1
            second_matched = self.localization_calls > 1
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=0,
                        matched=True,
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        confidence=0.94,
                    ),
                    MarkQuestionLocalizationItem(
                        mark_id=1,
                        matched=second_matched,
                        bbox=[0.55, 0.5, 0.8, 0.8] if second_matched else None,
                        confidence=0.94 if second_matched else 0.4,
                    ),
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            self.content_calls += 1
            returned_ids = [0] if self.content_calls == 1 else mark_ids
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=mark_id,
                        **_vision_result().items[mark_id].model_dump(),
                    )
                    for mark_id in returned_ids
                ],
                invalid_item_diagnostics=(
                    [
                        {
                            "mark_id": 1,
                            "validation_errors": [
                                {"field": "instruction", "type": "value_error"}
                            ],
                        }
                    ]
                    if self.content_calls == 1
                    else []
                ),
            )

    client = RetryClient()
    result, _localizations, _marks, diagnostic = recognize_marked_three_stage(
        client=client,
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        local_red_regions=[],
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        pair_max_distance_ratio=0.04,
        dedup_iou_threshold=0.8,
        crop_context_padding_ratio=0.1,
        image_max_edge=1200,
        image_jpeg_quality=90,
        image_max_pixels=40_000_000,
        localization_stage_retry_count=1,
        content_stage_retry_count=1,
        content_batch_size=6,
    )

    assert len(result.items) == 2
    assert client.localization_calls == 2
    assert client.content_calls == 2
    assert diagnostic["unlocalized_mark_ids"] == []
    assert diagnostic["missing_content_mark_ids"] == []
    assert diagnostic["content_invalid_item_count"] == 1
    assert diagnostic["content_invalid_mark_ids"] == [1]


def test_three_stage_keeps_unmatched_cross_as_pending_fallback(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
        recognize_marked_three_stage,
    )

    class CrossFallbackClient:
        def detect_marks(self, image_path, local_red_regions, correction=None):
            return MarkDetectionResult(error_marks=_vision_result().error_marks)

        def locate_marked_question_context(self, image_path, error_mark, correction=None):
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=error_mark.mark_id,
                        matched=True,
                        answer_bbox=[0.30, 0.30, 0.70, 0.70],
                        question_bbox=[0.10, 0.10, 0.90, 0.90],
                        confidence=0.94,
                    )
                ]
            )

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=error_marks[0].mark_id,
                        matched=False,
                        confidence=0.4,
                    )
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=mark_id,
                        **_vision_result().items[mark_id].model_dump(),
                    )
                    for mark_id in mark_ids
                ]
            )

    result, localizations, _marks, diagnostic = recognize_marked_three_stage(
        client=CrossFallbackClient(),
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        local_red_regions=[],
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        pair_max_distance_ratio=0.04,
        dedup_iou_threshold=0.8,
        crop_context_padding_ratio=0.1,
        image_max_edge=1200,
        image_jpeg_quality=90,
        image_max_pixels=40_000_000,
        localization_stage_retry_count=1,
    )

    assert len(result.items) == 2
    assert localizations[1].bbox_source == "cross_tolerant_fallback"
    assert localizations[1].localization_status == "needs_review"
    assert localizations[1].bbox[2] - localizations[1].bbox[0] >= 0.22 - 1e-6
    assert localizations[1].bbox[3] - localizations[1].bbox[1] >= 0.14 - 1e-6
    assert diagnostic["unlocalized_mark_ids"] == []


def test_three_stage_retries_mark_detection_for_uncovered_local_red_regions(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionItem,
        ContentRecognitionResult,
        MarkDetectionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
        recognize_marked_three_stage,
    )

    class MarkRetryClient:
        def __init__(self):
            self.mark_calls = 0

        def detect_marks(self, image_path, local_red_regions, correction=None):
            self.mark_calls += 1
            marks = _vision_result().error_marks
            return MarkDetectionResult(
                error_marks=marks[:1] if self.mark_calls == 1 else marks
            )

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=mark.mark_id,
                        matched=True,
                        bbox=(
                            [0.15, 0.15, 0.4, 0.45]
                            if mark.mark_id == 0
                            else [0.55, 0.5, 0.8, 0.8]
                        ),
                        confidence=0.94,
                    )
                    for mark in error_marks
                ]
            )

        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            return ContentRecognitionResult(
                items=[
                    ContentRecognitionItem(
                        mark_id=mark_id,
                        **_vision_result().items[mark_id].model_dump(),
                    )
                    for mark_id in mark_ids
                ]
            )

    client = MarkRetryClient()
    result, _localizations, _marks, diagnostic = recognize_marked_three_stage(
        client=client,
        image_path=_write_source_image(tmp_path),
        subject_hint="chinese",
        local_red_regions=[[0.2, 0.23, 0.3, 0.34], [0.62, 0.6, 0.73, 0.72]],
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        red_pixel_expansion_ratio=0.05,
        pair_max_distance_ratio=0.04,
        dedup_iou_threshold=0.8,
        crop_context_padding_ratio=0.1,
        image_max_edge=1200,
        image_jpeg_quality=90,
        image_max_pixels=40_000_000,
        mark_stage_retry_count=1,
    )

    assert client.mark_calls == 2
    assert len(result.items) == 2
    assert diagnostic["uncovered_local_red_region_count"] == 0


def test_marked_mode_localizes_normalized_error_mark_groups(tmp_path):
    class DuplicateMarkClient(FakeClient):
        def recognize(self, image_path, subject_hint=None, **_kwargs):
            self.recognize_calls += 1
            result = _vision_result()
            result.items = result.items[:1]
            result.error_marks = [
                ErrorMark(
                    mark_id=0,
                    mark_type="circle",
                    bbox=[0.2, 0.23, 0.3, 0.34],
                    confidence=0.96,
                ),
                ErrorMark(
                    mark_id=1,
                    mark_type="circle",
                    bbox=[0.2, 0.23, 0.3, 0.34],
                    confidence=0.95,
                ),
            ]
            return result

        def localize(self, image_path, items, error_marks):
            self.localize_calls += 1
            self.localized_marks = error_marks
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=0,
                        matched=True,
                        mark_ids=[0],
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        observed_prompt_text="课文",
                        observed_raw_text="kè wén",
                        confidence=0.94,
                    )
                ]
            )

    client = DuplicateMarkClient()
    _result, values = _run_batch(
        tmp_path,
        client=client,
        local_red_scan=_detected_red_scan(),
        correction_group_enabled=True,
    )

    assert [mark.mark_id for mark in client.localized_marks] == [0]
    assert values[0]["ocr_raw_json"]["correction_group_validation"] == {
        "raw_mark_count": 2,
        "correction_group_count": 1,
        "paired_group_count": 0,
            "single_mark_group_count": 1,
            "deduplicated_mark_count": 1,
            "review_required_mark_ids": [],
        }


def test_semantic_localization_retry_is_skipped_when_first_result_is_valid(tmp_path):
    client = FakeClient()

    _run_batch(
        tmp_path,
        client=client,
        correction_group_enabled=True,
        semantic_retry_count=1,
    )

    assert client.localize_calls == 1


def test_semantic_localization_retry_replaces_wrong_mark_assignment(tmp_path):
    class SemanticRetryClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.corrections = []

        def localize(self, image_path, items, error_marks, correction=None):
            self.localize_calls += 1
            self.corrections.append(correction)
            mark_ids = ([1], [0]) if self.localize_calls == 1 else ([0], [1])
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=0,
                        matched=True,
                        mark_ids=mark_ids[0],
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        observed_prompt_text=items[0].prompt_text,
                        observed_raw_text=items[0].raw_text,
                        confidence=0.94,
                    ),
                    LocalizationItem(
                        index=1,
                        matched=True,
                        mark_ids=mark_ids[1],
                        bbox=[0.55, 0.5, 0.8, 0.8],
                        observed_prompt_text=items[1].prompt_text,
                        observed_raw_text=items[1].raw_text,
                        confidence=0.91,
                    ),
                ]
            )

    client = SemanticRetryClient()
    _result, values = _run_batch(
        tmp_path,
        client=client,
        correction_group_enabled=True,
        semantic_retry_count=1,
    )

    assert client.localize_calls == 2
    assert client.corrections[0] is None
    assert client.corrections[1]["reason_counts"] == {"mark_anchor_too_far": 2}
    assert values[0]["crop_region"]["mark_ids"] == [0]
    assert values[1]["crop_region"]["mark_ids"] == [1]
    batch = values[0]["ocr_raw_json"]["localization_batch_validation"]
    assert batch["semantic_retry_attempts"] == 1
    assert batch["semantic_retry_reason_counts"] == {"mark_anchor_too_far": 2}


def test_marked_ocr_rescue_is_bounded_for_deterministic_assignments(tmp_path):
    class MissingAssignmentsClient(FakeClient):
        def localize(self, image_path, items, error_marks):
            self.localize_calls += 1
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=0,
                        matched=True,
                        mark_ids=[],
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        observed_prompt_text=items[0].prompt_text,
                        observed_raw_text=items[0].raw_text,
                        confidence=0.94,
                    ),
                    LocalizationItem(
                        index=1,
                        matched=True,
                        mark_ids=[],
                        bbox=[0.55, 0.5, 0.8, 0.8],
                        observed_prompt_text=items[1].prompt_text,
                        observed_raw_text=items[1].raw_text,
                        confidence=0.91,
                    ),
                ]
            )

    verifier = FakeOCRVerifier(
        {
            0: OCRVerification(status="support"),
            1: OCRVerification(status="support"),
        }
    )
    _result, values = _run_batch(
        tmp_path,
        client=MissingAssignmentsClient(),
        ocr_verifier=verifier,
        correction_group_enabled=True,
        marked_ocr_recheck_limit=1,
    )

    assert len(verifier.calls) == 1
    assert sum(
        value["ocr_raw_json"]["assignment_source"] == "deterministic"
        for value in values
    ) == 2
    assert values[0]["ocr_raw_json"]["localization_batch_validation"][
        "marked_ocr_recheck_count"
    ] == 1


def test_marked_text_mismatch_is_sent_to_ocr_and_retained_for_review(tmp_path):
    class TextMismatchClient(FakeClient):
        def localize(self, image_path, items, error_marks):
            self.localize_calls += 1
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=0,
                        matched=True,
                        mark_ids=[0],
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        observed_prompt_text="错误文本",
                        observed_raw_text="错误文本",
                        confidence=0.94,
                    ),
                    LocalizationItem(
                        index=1,
                        matched=True,
                        mark_ids=[1],
                        bbox=[0.55, 0.5, 0.8, 0.8],
                        observed_prompt_text=items[1].prompt_text,
                        observed_raw_text=items[1].raw_text,
                        confidence=0.91,
                    ),
                ]
            )

    verifier = FakeOCRVerifier({0: OCRVerification(status="support")})
    _result, values = _run_batch(
        tmp_path,
        client=TextMismatchClient(),
        ocr_verifier=verifier,
        local_red_scan=_detected_red_scan(),
        correction_group_enabled=True,
        marked_ocr_recheck_limit=1,
    )

    assert verifier.calls[0][1] == 0
    assert values[0]["collection_status"] == "pending_review"
    assert values[0]["crop_region"]["localization_status"] == "verified"
    validation = values[0]["ocr_raw_json"]["localization_validation"]
    assert validation["text_evidence_passed"] is False
    assert validation["ocr_text_rescued"] is True
    assert values[0]["ocr_raw_json"]["reliable_error_mark"] is True


def test_uncovered_local_red_region_uniquely_rescues_missing_mark_assignment(tmp_path):
    from app.services.error_mark_validation import RedMarkRegion, RedMarkScanResult

    class OneMarkClient(FakeClient):
        def recognize(self, image_path, subject_hint=None, **_kwargs):
            result = _vision_result()
            result.error_marks = result.error_marks[:1]
            return result

        def localize(self, image_path, items, error_marks):
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=0,
                        matched=True,
                        mark_ids=[0],
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        observed_prompt_text=items[0].prompt_text,
                        observed_raw_text=items[0].raw_text,
                        confidence=0.94,
                    ),
                    LocalizationItem(
                        index=1,
                        matched=True,
                        mark_ids=[],
                        bbox=[0.55, 0.5, 0.8, 0.8],
                        observed_prompt_text=items[1].prompt_text,
                        observed_raw_text=items[1].raw_text,
                        confidence=0.91,
                    ),
                ]
            )

    scan = RedMarkScanResult(
        status="detected",
        regions=[
            RedMarkRegion(
                bbox=[0.2, 0.23, 0.3, 0.34],
                pixel_count=100,
                area_ratio=0.011,
                thinness_ratio=1.1,
            ),
            RedMarkRegion(
                bbox=[0.62, 0.6, 0.73, 0.72],
                pixel_count=700,
                area_ratio=0.013,
                thinness_ratio=1.1,
            ),
        ],
        red_pixel_count=800,
        scanned_width=400,
        scanned_height=300,
        duration_ms=1.0,
    )

    _result, values = _run_batch(
        tmp_path,
        client=OneMarkClient(),
        local_red_scan=scan,
        correction_group_enabled=True,
        local_red_rescue_min_pixels=80,
        marked_ocr_recheck_limit=1,
    )

    assert values[1]["collection_status"] == "pending_review"
    assert values[1]["ocr_raw_json"]["assignment_source"] == "local_red"
    assert values[1]["ocr_raw_json"]["localization_red_validation"]["accepted"]
    batch = values[0]["ocr_raw_json"]["localization_batch_validation"]
    assert batch["local_red_rescue_indexes"] == [1]
    assert batch["local_red_rescue_diagnostics"][0]["region_pixel_count"] == 700


def test_marked_mode_retries_once_when_model_misses_local_red_mark(tmp_path):
    class RetryClient(FakeClient):
        def recognize(self, image_path, subject_hint=None, **_kwargs):
            self.recognize_calls += 1
            result = _vision_result()
            if self.recognize_calls == 1:
                result.error_marks = []
            return result

    client = RetryClient()
    _result, values = _run_batch(
        tmp_path,
        client=client,
        local_red_scan=_detected_red_scan(),
        mark_mismatch_retry_count=1,
    )

    assert client.recognize_calls == 2
    assert values


def test_marked_mode_returns_image_issue_after_retry_exhaustion(tmp_path):
    from app.services.vision_recognition import ImageReviewRequired

    class MissingMarkClient(FakeClient):
        def recognize(self, image_path, subject_hint=None, **_kwargs):
            self.recognize_calls += 1
            result = _vision_result()
            result.error_marks = []
            return result

    client = MissingMarkClient()
    with pytest.raises(ImageReviewRequired) as raised:
        _run_batch(
            tmp_path,
            client=client,
            local_red_scan=_detected_red_scan(),
            mark_mismatch_retry_count=1,
        )

    assert raised.value.code == "red_marks_unresolved"
    assert client.recognize_calls == 2


def test_marked_mode_returns_image_issue_when_localization_discards_every_candidate(
    tmp_path,
):
    from app.services.vision_recognition import ImageReviewRequired

    class RejectedLocalizationClient(FakeClient):
        def localize(self, image_path, items, error_marks):
            self.localize_calls += 1
            raise VisionRecognitionError(
                "vision_localization_invalid",
                "题目定位结果不完整，请稍后重试",
                diagnostic={
                    "operation": "localization",
                    "reason": "unassigned_mark",
                    "candidate_count": len(items),
                    "mark_count": len(error_marks),
                },
            )

    with pytest.raises(ImageReviewRequired) as raised:
        _run_batch(
            tmp_path,
            client=RejectedLocalizationClient(),
            local_red_scan=_detected_red_scan(),
        )

    assert raised.value.code == "red_marks_unresolved"
    assert raised.value.diagnostic == {
        "operation": "localization",
        "reason": "marked_candidates_without_reliable_localization",
        "candidate_count": 2,
        "mark_count": 2,
        "localization_status": "rejected",
        "localization_error_code": "vision_localization_invalid",
        "localization_error_reason": "unassigned_mark",
        "localization_error_diagnostic": {
            "operation": "localization",
            "reason": "unassigned_mark",
            "candidate_count": 2,
            "mark_count": 2,
        },
        "localization_returned_count": 0,
        "localization_validated_count": 0,
        "localization_verified_count": 0,
        "localization_reliable_mark_count": 0,
        "localization_rejection_counts": {
            "present": 2,
            "matched": 2,
            "has_bbox": 2,
            "confidence_passed": 2,
            "geometry_passed": 2,
            "text_evidence_passed": 2,
            "verified": 2,
        },
    }
    from app.services.vision_recognition import safe_recognition_diagnostic

    assert safe_recognition_diagnostic(raised.value) == raised.value.diagnostic


def test_marked_mode_reports_specific_geometry_rejection_reasons(tmp_path):
    from app.services.vision_recognition import ImageReviewRequired

    class GeometryRejectedClient(FakeClient):
        def localize(self, image_path, items, error_marks):
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=0,
                        matched=True,
                        mark_ids=[0],
                        bbox=[0.0, 0.0, 1.0, 0.8],
                        observed_prompt_text=items[0].prompt_text,
                        observed_raw_text=items[0].raw_text,
                        confidence=0.95,
                    ),
                    LocalizationItem(
                        index=1,
                        matched=True,
                        mark_ids=[1],
                        bbox=[0.8, 0.8, 0.95, 0.95],
                        observed_prompt_text=items[1].prompt_text,
                        observed_raw_text=items[1].raw_text,
                        confidence=0.95,
                    ),
                ]
            )

    with pytest.raises(ImageReviewRequired) as raised:
        _run_batch(
            tmp_path,
            client=GeometryRejectedClient(),
            local_red_scan=_detected_red_scan(),
        )

    assert raised.value.diagnostic["localization_geometry_failure_counts"] == {
        "bbox_area_exceeded": 1,
        "mark_center_outside_bbox": 1,
    }
    assert raised.value.diagnostic["localization_geometry_diagnostics"] == [
        {
            "index": 0,
            "passed": False,
            "bbox_area_ratio": 0.8,
            "max_area_ratio": 0.35,
            "mark_ids": [0],
            "missing_mark_ids": [],
            "outside_mark_ids": [],
            "outside_mark_diagnostics": [],
            "failure_reasons": ["bbox_area_exceeded"],
        },
        {
            "index": 1,
            "passed": False,
            "bbox_area_ratio": 0.0225,
            "max_area_ratio": 0.35,
            "mark_ids": [1],
            "missing_mark_ids": [],
            "outside_mark_ids": [1],
            "outside_mark_diagnostics": [
                {
                    "mark_id": 1,
                    "horizontal_gap_ratio": 0.125,
                    "vertical_gap_ratio": 0.14,
                    "nearest_distance_ratio": 0.187683,
                    "mark_bbox_intersects_question_bbox": False,
                }
            ],
            "failure_reasons": ["mark_center_outside_bbox"],
        },
    ]


def test_marked_mode_keeps_reliable_candidates_when_a_mark_is_unassigned(tmp_path):
    class PartiallyAssignedClient(FakeClient):
        def localize(self, image_path, items, error_marks):
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=0,
                        matched=True,
                        mark_ids=[0],
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        observed_prompt_text=items[0].prompt_text,
                        observed_raw_text=items[0].raw_text,
                        confidence=0.95,
                    ),
                    LocalizationItem(
                        index=1,
                        matched=True,
                        mark_ids=[],
                        bbox=[0.55, 0.5, 0.8, 0.8],
                        observed_prompt_text=items[1].prompt_text,
                        observed_raw_text=items[1].raw_text,
                        confidence=0.95,
                    ),
                ]
            )

    _result, values = _run_batch(
        tmp_path,
        client=PartiallyAssignedClient(),
        local_red_scan=_detected_red_scan(),
    )

    batch_diagnostic = values[0]["ocr_raw_json"][
        "localization_batch_validation"
    ]
    assert batch_diagnostic["status"] == "validated"
    assert batch_diagnostic["assigned_mark_ids"] == [0]
    assert batch_diagnostic["unassigned_mark_ids"] == [1]
    assert batch_diagnostic["unassigned_mark_count"] == 1
    assert len(batch_diagnostic["missing_mark_diagnostics"]) == 1
    missing_mark_diagnostic = batch_diagnostic["missing_mark_diagnostics"][0]
    assert missing_mark_diagnostic["index"] == 1
    assert missing_mark_diagnostic["local_red_validation"]["accepted"] is True
    assert missing_mark_diagnostic["local_red_validation"]["reason"] == "accepted"
    assert missing_mark_diagnostic["nearest_local_red_region"] == {
        "region_index": 0,
        "horizontal_gap_ratio": 0.25,
        "vertical_gap_ratio": 0.16,
        "nearest_distance_ratio": 0.296816,
        "intersects_question_bbox": False,
        "region_area_ratio": 0.011,
        "region_pixel_count": 100,
    }
    assert values[0]["collection_status"] == "pending_review"
    assert values[1]["collection_status"] == "pending_review"


def test_unmarked_mode_uses_one_page_ocr_and_at_most_three_crop_rechecks(tmp_path):
    from app.services.error_mark_validation import RedMarkScanResult
    from app.services.local_ocr_verification import OCRPageEvidence

    items = [
        _vision_result().items[0].model_copy(
            update={
                "prompt_text": f"提示{index}",
                "raw_text": f"作答{index}",
                "answer": f"作答{index}",
                "confidence": 0.99 - index / 1000,
            }
        )
        for index in range(20)
    ]

    class UnmarkedClient(FakeClient):
        def recognize(self, image_path, subject_hint=None, **_kwargs):
            return VisionResult(items=items, error_marks=[], ignored_text=[])

        def localize(self, image_path, recognized_items, error_marks):
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=index,
                        matched=True,
                        mark_ids=[],
                        bbox=[0.05, 0.02 + index * 0.045, 0.95, 0.055 + index * 0.045],
                        observed_prompt_text=item.prompt_text,
                        observed_raw_text=item.raw_text,
                        confidence=0.95,
                    )
                    for index, item in enumerate(recognized_items)
                ]
            )

    class CountingOCR(FakeOCRVerifier):
        enabled = True
        line_confidence_threshold = 0.85
        min_effective_characters = 2
        support_similarity_threshold = 0.8
        contradiction_similarity_threshold = 0.9

        def __init__(self):
            super().__init__()
            self.page_calls = 0
            self.crop_calls = 0

        def recognize_page(self, image_path, max_edge):
            self.page_calls += 1
            return OCRPageEvidence(
                status="available",
                lines=[],
                duration_ms=1.0,
                prepared_size=[400, 300],
            )

        def verify_crop(self, image_path, bbox, target_index, items):
            self.crop_calls += 1
            return OCRVerification(status="inconclusive")

    verifier = CountingOCR()
    no_red = RedMarkScanResult(
        status="none",
        regions=[],
        red_pixel_count=0,
        scanned_width=400,
        scanned_height=300,
        duration_ms=1.0,
    )
    _run_batch(
        tmp_path,
        client=UnmarkedClient(),
        ocr_verifier=verifier,
        local_red_scan=no_red,
    )

    assert verifier.page_calls == 1
    assert verifier.crop_calls == 3


def test_pipeline_recognizes_and_localizes_once_per_image(tmp_path):
    client = FakeClient()
    result, values = _run_batch(tmp_path, client=client)

    assert result.items[0].prompt_text == "课文"
    assert client.recognize_calls == 1
    assert client.localize_calls == 1
    assert [mark.mark_id for mark in client.localized_marks] == [0, 1]
    assert values[0]["tags"] == ["拼音", "老师批改"]
    assert values[1]["tags"] == ["词语", "错别字"]
    assert values[0]["crop_region"]["bbox"] == [0.15, 0.15, 0.4, 0.45]


def test_pipeline_falls_back_without_candidate_bbox_when_localization_fails(tmp_path):
    client = FakeClient(localization_error=True)
    _result, values = _run_batch(tmp_path, client=client)

    assert client.localize_calls == 1
    assert all("bbox" not in value["crop_region"] for value in values)
    assert all(value["review_status"] == "needs_review" for value in values)


def test_pipeline_rejects_model_marks_without_local_red_pixels(tmp_path):
    class InvalidMarkClient(FakeClient):
        def recognize(self, image_path, subject_hint=None):
            result = _vision_result()
            result.error_marks = [
                mark.model_copy(update={"bbox": [0.8, 0.05, 0.9, 0.15]})
                for mark in result.error_marks
            ]
            return result

    client = InvalidMarkClient()
    _result, values = _run_batch(tmp_path, client=client)

    assert client.localize_calls == 0
    assert all("bbox" not in value["crop_region"] for value in values)
    assert all(value["review_status"] == "needs_review" for value in values)


def test_unmarked_mode_localizes_when_model_marks_are_rejected(tmp_path):
    from app.services.error_mark_validation import RedMarkScanResult

    class InvalidMarkUnmarkedClient(FakeClient):
        def recognize(self, image_path, subject_hint=None, **_kwargs):
            result = _vision_result()
            result.error_marks = [
                result.error_marks[0].model_copy(
                    update={"bbox": [0.8, 0.05, 0.9, 0.15]}
                )
            ]
            return result

        def localize(self, image_path, items, error_marks):
            self.localize_calls += 1
            assert error_marks == []
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=index,
                        matched=True,
                        mark_ids=[],
                        bbox=[0.1 + index * 0.4, 0.15, 0.35 + index * 0.4, 0.45],
                        observed_prompt_text=item.prompt_text,
                        observed_raw_text=item.raw_text,
                        confidence=0.95,
                    )
                    for index, item in enumerate(items)
                ]
            )

    no_red = RedMarkScanResult(
        status="none",
        regions=[],
        red_pixel_count=0,
        scanned_width=400,
        scanned_height=300,
        duration_ms=1.0,
    )
    client = InvalidMarkUnmarkedClient()

    _result, values = _run_batch(
        tmp_path,
        client=client,
        local_red_scan=no_red,
        force_mode="unmarked",
    )

    assert client.localize_calls == 1
    assert all("bbox" in value["crop_region"] for value in values)


def test_rejected_mark_does_not_invalidate_question_matched_to_valid_mark(tmp_path):
    class PartiallyValidMarkClient(FakeClient):
        def recognize(self, image_path, subject_hint=None):
            result = _vision_result()
            result.error_marks[1] = result.error_marks[1].model_copy(
                update={"bbox": [0.8, 0.05, 0.9, 0.15]}
            )
            return result

        def localize(self, image_path, items, error_marks):
            self.localize_calls += 1
            self.localized_marks = error_marks
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=0,
                        matched=True,
                        mark_ids=[0],
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        observed_prompt_text=items[0].prompt_text,
                        observed_raw_text=items[0].raw_text,
                        confidence=0.94,
                    ),
                    LocalizationItem(
                        index=1,
                        matched=False,
                        mark_ids=[],
                        bbox=None,
                        observed_prompt_text=None,
                        observed_raw_text=None,
                        confidence=0.0,
                    ),
                ]
            )

    verifier = FakeOCRVerifier()
    client = PartiallyValidMarkClient()
    _result, values = _run_batch(
        tmp_path,
        client=client,
        ocr_verifier=verifier,
    )

    assert [mark.mark_id for mark in client.localized_marks] == [0]
    assert values[0]["crop_region"]["bbox"] == [0.15, 0.15, 0.4, 0.45]
    assert values[0]["review_status"] == "confirmed"
    assert verifier.calls == [([0.15, 0.15, 0.4, 0.45], 0)]
    assert "bbox" not in values[1]["crop_region"]
    assert values[1]["review_status"] == "needs_review"


def test_empty_mark_ids_use_local_red_evidence_for_trusted_localization(tmp_path):
    class MissingMarkAnchorClient(FakeClient):
        def recognize(self, image_path, subject_hint=None):
            result = _vision_result()
            result.error_marks[1] = result.error_marks[1].model_copy(
                update={"bbox": [0.8, 0.05, 0.9, 0.15]}
            )
            return result

        def localize(self, image_path, items, error_marks):
            self.localize_calls += 1
            self.localized_marks = error_marks
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=0,
                        matched=True,
                        mark_ids=[0],
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        observed_prompt_text=items[0].prompt_text,
                        observed_raw_text=items[0].raw_text,
                        confidence=0.94,
                    ),
                    LocalizationItem(
                        index=1,
                        matched=True,
                        mark_ids=[],
                        bbox=[0.55, 0.5, 0.8, 0.8],
                        observed_prompt_text=items[1].prompt_text,
                        observed_raw_text=items[1].raw_text,
                        confidence=0.91,
                    ),
                ]
            )

    verifier = FakeOCRVerifier()
    _result, values = _run_batch(
        tmp_path,
        client=MissingMarkAnchorClient(),
        ocr_verifier=verifier,
    )

    assert values[1]["review_status"] == "confirmed"
    assert values[1]["crop_region"]["bbox"] == [0.55, 0.5, 0.8, 0.8]
    assert values[1]["crop_region"]["bbox_source"] == "local_red_verified"
    assert values[1]["crop_region"]["mark_ids"] == []
    assert values[1]["ocr_raw_json"]["localization_red_validation"]["accepted"] is True
    assert verifier.calls[-1] == ([0.55, 0.5, 0.8, 0.8], 1)


def test_empty_mark_ids_without_local_red_evidence_still_need_review(tmp_path):
    class MissingMarkAndRedClient(FakeClient):
        def recognize(self, image_path, subject_hint=None):
            result = _vision_result()
            result.error_marks[1] = result.error_marks[1].model_copy(
                update={"bbox": [0.8, 0.05, 0.9, 0.15]}
            )
            return result

        def localize(self, image_path, items, error_marks):
            self.localize_calls += 1
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=0,
                        matched=True,
                        mark_ids=[0],
                        bbox=[0.15, 0.15, 0.4, 0.45],
                        observed_prompt_text=items[0].prompt_text,
                        observed_raw_text=items[0].raw_text,
                        confidence=0.94,
                    ),
                    LocalizationItem(
                        index=1,
                        matched=True,
                        mark_ids=[],
                        bbox=[0.75, 0.75, 0.95, 0.95],
                        observed_prompt_text=items[1].prompt_text,
                        observed_raw_text=items[1].raw_text,
                        confidence=0.91,
                    ),
                ]
            )

    _result, values = _run_batch(
        tmp_path,
        client=MissingMarkAndRedClient(),
    )

    assert values[1]["review_status"] == "needs_review"
    assert "bbox" not in values[1]["crop_region"]
    diagnostic = values[1]["ocr_raw_json"]["localization_red_validation"]
    assert diagnostic["accepted"] is False
    assert diagnostic["reason"] == "insufficient_red_pixels"


def test_ocr_contradiction_discards_localized_bbox(tmp_path):
    verifier = FakeOCRVerifier(
        {
            0: OCRVerification(
                status="wrong_candidate",
                matched_index=1,
                text_summary="算式",
                confidence=0.98,
            )
        }
    )

    _result, values = _run_batch(tmp_path, ocr_verifier=verifier)

    assert "bbox" not in values[0]["crop_region"]
    assert values[0]["review_status"] == "needs_review"


def test_ocr_inconclusive_keeps_otherwise_valid_bbox(tmp_path):
    _result, values = _run_batch(tmp_path, ocr_verifier=FakeOCRVerifier())

    assert values[0]["crop_region"]["bbox"] == [0.15, 0.15, 0.4, 0.45]
    assert values[0]["review_status"] == "confirmed"


def test_pipeline_expands_display_bbox_but_ocr_uses_localization_bbox(tmp_path):
    verifier = FakeOCRVerifier()

    _result, values = _run_batch(
        tmp_path,
        ocr_verifier=verifier,
        crop_context_padding_ratio=0.15,
    )

    assert values[0]["crop_region"]["bbox"] == pytest.approx(
        [0.0875, 0.09, 0.4125, 0.48]
    )
    assert values[0]["crop_region"]["localization_bbox"] == [
        0.15,
        0.15,
        0.4,
        0.45,
    ]
    assert values[0]["crop_region"]["display_context_padding_ratio"] == 0.15
    assert verifier.calls[0] == ([0.15, 0.15, 0.4, 0.45], 0)


def test_saved_diagnostics_separate_marks_localization_and_ocr(tmp_path):
    verifier = FakeOCRVerifier(
        {
            0: OCRVerification(
                status="support",
                matched_index=0,
                text_summary="课文",
                confidence=0.98,
            )
        }
    )

    _result, values = _run_batch(tmp_path, ocr_verifier=verifier)

    raw = values[0]["ocr_raw_json"]
    assert raw["error_marks"][0]["mark_id"] == 0
    assert raw["error_mark_validation"][0]["mark_id"] == 0
    assert raw["error_mark_validation"][0]["accepted"] is True
    assert raw["error_mark_validation"][0]["red_pixel_ratio"] >= 0.01
    assert raw["localization"]["mark_ids"] == [0]
    assert raw["local_ocr"]["status"] == "support"
