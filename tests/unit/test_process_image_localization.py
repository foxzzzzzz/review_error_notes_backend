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
    evidence_mode=False,
    image_id=None,
    deadline=None,
    ocr_full_page_max_edge=1600,
    ocr_crop_recheck_limit=3,
    evidence_ocr_crop_recheck_limit=None,
    localization_stage_retry_count=1,
    evidence_localization_recheck_limit=None,
    stage_audit_enabled=False,
):
    from app.services.vision_recognition import recognize_question_batch

    evidence_kwargs = {}
    if (
        evidence_mode
        or image_id is not None
        or deadline is not None
        or evidence_localization_recheck_limit is not None
    ):
        evidence_kwargs = {
            "evidence_mode": evidence_mode,
            "image_id": image_id,
            "deadline": deadline,
            "stage_audit_enabled": stage_audit_enabled,
        }
        if evidence_ocr_crop_recheck_limit is not None:
            evidence_kwargs["evidence_ocr_crop_recheck_limit"] = (
                evidence_ocr_crop_recheck_limit
            )
        if evidence_localization_recheck_limit is not None:
            evidence_kwargs["evidence_localization_recheck_limit"] = (
                evidence_localization_recheck_limit
            )
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
        localization_stage_retry_count=localization_stage_retry_count,
        ocr_full_page_max_edge=ocr_full_page_max_edge,
        ocr_crop_recheck_limit=ocr_crop_recheck_limit,
        **evidence_kwargs,
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


def _evidence_content_item(
    mark_id,
    *,
    raw_text="学生原文",
    answer=None,
    printed_prompt=None,
    printed_pinyin=None,
    student_handwriting=None,
):
    from app.services.vision_recognition import ContentRecognitionItem

    return ContentRecognitionItem(
        mark_id=mark_id,
        raw_text=raw_text,
        instruction="看拼音写词语",
        prompt_text="题目提示",
        normalized_text=None,
        answer=answer,
        subject="chinese",
        question_type="write_word",
        tags=["词语"],
        difficulty=2,
        confidence=0.9,
        uncertain_segments=[],
        printed_prompt=printed_prompt,
        printed_pinyin=printed_pinyin,
        student_handwriting=student_handwriting,
    )


class _EvidenceStageClient:
    def __init__(self, *, content_result=None, content_error=None):
        self.content_result = content_result
        self.content_error = content_error
        self.content_calls = 0

    def detect_marks(self, image_path, local_red_regions, correction=None):
        from app.services.vision_recognition import MarkDetectionResult

        return MarkDetectionResult(error_marks=_vision_result().error_marks)

    def locate_marked_questions(self, image_path, error_marks, correction=None):
        from app.services.vision_recognition import (
            MarkQuestionLocalizationItem,
            MarkQuestionLocalizationResult,
        )

        geometries = {
            0: {
                "bbox": [0.1, 0.1, 0.4, 0.45],
                "answer_bbox": [0.2, 0.25, 0.3, 0.35],
                "prompt_bbox": [0.12, 0.12, 0.38, 0.22],
            },
            1: {
                "bbox": [0.55, 0.5, 0.8, 0.8],
                "answer_bbox": [0.62, 0.62, 0.72, 0.72],
                "prompt_bbox": [0.57, 0.52, 0.78, 0.6],
            },
        }
        return MarkQuestionLocalizationResult(
            items=[
                MarkQuestionLocalizationItem(
                    mark_id=mark.mark_id,
                    matched=True,
                    confidence=0.94,
                    **geometries[mark.mark_id],
                )
                for mark in error_marks
            ]
        )

    def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
        self.content_calls += 1
        if self.content_error is not None:
            raise self.content_error
        return self.content_result


def _run_evidence_three_stage(
    tmp_path,
    client,
    *,
    ocr_page_evidence=None,
    image_id="image-7",
    deadline=None,
    evidence_mode=True,
    content_batch_size=6,
    mark_stage_retry_count=0,
    localization_stage_retry_count=0,
    local_red_evidence_regions=None,
    stage_audit_enabled=False,
):
    from app.services.vision_recognition import recognize_marked_three_stage

    return recognize_marked_three_stage(
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
        mark_stage_retry_count=mark_stage_retry_count,
        localization_stage_retry_count=localization_stage_retry_count,
        content_stage_retry_count=0,
        content_batch_size=content_batch_size,
        evidence_mode=evidence_mode,
        image_id=image_id,
        ocr_page_evidence=ocr_page_evidence,
        deadline=deadline,
        local_red_evidence_regions=local_red_evidence_regions,
        stage_audit_enabled=stage_audit_enabled,
    )


def test_three_stage_audit_preserves_red_components_and_full_page_ocr(tmp_path):
    from app.services.error_mark_validation import RedMarkRegion
    from app.services.local_ocr_verification import OCRLine, OCRPageEvidence
    from app.services.vision_recognition import ContentRecognitionResult

    client = _EvidenceStageClient(
        content_result=ContentRecognitionResult(
            items=[_evidence_content_item(0), _evidence_content_item(1)]
        )
    )
    red_region = RedMarkRegion(
        bbox=[0.2, 0.2, 0.3, 0.35],
        pixel_count=123,
        area_ratio=0.015,
        thinness_ratio=1.5,
    )
    ocr_page = OCRPageEvidence(
        status="available",
        lines=[
            OCRLine(
                text="看拼音写词语",
                confidence=0.96,
                bbox=[0.1, 0.1, 0.4, 0.2],
            )
        ],
        duration_ms=4.5,
        prepared_size=[400, 300],
    )

    _values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path,
        client,
        ocr_page_evidence=ocr_page,
        local_red_evidence_regions=[red_region],
        stage_audit_enabled=True,
    )

    audit = diagnostic["stage_audit"]
    assert audit["red_components"] == [
        {
            "component_id": 0,
            "bbox": [0.2, 0.2, 0.3, 0.35],
            "pixel_count": 123,
            "area_ratio": 0.015,
            "thinness_ratio": 1.5,
        }
    ]
    assert audit["ocr_lines"] == [
        {
            "ocr_line_id": 0,
            "text": "看拼音写词语",
            "confidence": 0.96,
            "bbox": [0.1, 0.1, 0.4, 0.2],
        }
    ]
    assert audit["mark_attempts"][0]["entry"] == "direct"
    assert audit["mark_attempts"][0]["attempt"] == 1
    assert audit["merged_primitives"][0]["source_refs"] == ["attempt-1:0"]
    assert audit["mark_events"][0]["member_merged_mark_ids"] == [0]


def test_stage_audit_composite_preserves_absorbed_cross_and_circle_sources():
    from app.services.vision_recognition import (
        ErrorMark,
        _audit_merged_mark_sources,
    )

    composite = ErrorMark(
        mark_id=0,
        mark_type="cross_circle",
        bbox=[0.1, 0.1, 0.4, 0.4],
        cross_bbox=[0.2, 0.2, 0.3, 0.3],
        circle_bbox=[0.1, 0.1, 0.4, 0.4],
        confidence=0.95,
    )
    attempt_marks = [
        {
            "source_ref": "attempt-1:0",
            "mark_type": "cross_circle",
            "bbox": [0.1, 0.1, 0.4, 0.4],
            "cross_bbox": [0.2, 0.2, 0.3, 0.3],
            "circle_bbox": [0.1, 0.1, 0.4, 0.4],
        },
        {
            "source_ref": "attempt-2:0",
            "mark_type": "cross",
            "bbox": [0.2, 0.2, 0.3, 0.3],
        },
        {
            "source_ref": "attempt-2:1",
            "mark_type": "circle",
            "bbox": [0.1, 0.1, 0.4, 0.4],
        },
    ]

    records = _audit_merged_mark_sources(
        [composite], attempt_marks, dedup_iou_threshold=0.8
    )

    assert records[0]["source_refs"] == [
        "attempt-1:0",
        "attempt-2:0",
        "attempt-2:1",
    ]


def _real_deepseek_transport_client(*, localization_item, content_item):
    import json

    import httpx

    from app.services.deepseek_vision import DeepSeekVisionClient

    responses = iter(
        [
            {
                "error_marks": [
                    {
                        "mark_id": 0,
                        "mark_type": "cross",
                        "bbox": [0.2, 0.23, 0.3, 0.34],
                        "cross_bbox": None,
                        "circle_bbox": None,
                        "confidence": 0.96,
                    }
                ]
            },
            {"items": [localization_item]},
            {"items": [content_item]},
        ]
    )

    def handler(_request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(next(responses), ensure_ascii=False)
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    return DeepSeekVisionClient(
        api_key="test-key",
        api_host="https://deepseek.invalid/v1",
        timeout_seconds=1,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
        model="deepseek-test",
        thinking="disabled",
        max_tokens=1024,
        image_detail="original",
    )


def test_evidence_mode_all_invalid_content_conserves_each_localized_mark(tmp_path):
    from app.services.vision_recognition import ContentRecognitionResult

    client = _EvidenceStageClient(
        content_result=ContentRecognitionResult(
            items=[],
            invalid_item_diagnostics=[
                {"mark_id": 0, "validation_errors": [{"field": "instruction"}]},
                {"mark_id": 1, "validation_errors": [{"field": "prompt_text"}]},
            ],
        )
    )

    values, localizations, marks, diagnostic = _run_evidence_three_stage(
        tmp_path, client
    )

    assert [value["ocr_raw_json"]["evidence_bundle"]["identity"]["mark_id"] for value in values] == [0, 1]
    assert len(values) == len(localizations) == len(marks) == 2
    assert all(value["collection_status"] == "pending_review" for value in values)
    assert all(value["answer_status"] == "unresolved" for value in values)
    assert all(value["question_evidence_status"] == "insufficient" for value in values)
    assert all(value["subject"] == "chinese" for value in values)
    assert values[0]["ocr_raw_json"]["content_diagnostic"] == {
        "status": "invalid",
        "invalid_items": [
            {
                "mark_id": 0,
                "validation_errors": [{"field": "instruction"}],
                "content_attempt": 1,
            }
        ],
    }
    assert diagnostic["missing_content_mark_ids"] == [0, 1]


@pytest.mark.parametrize("error_code", ["vision_service_unavailable", "vision_timeout"])
def test_evidence_mode_content_failure_becomes_safe_placeholders(tmp_path, error_code):
    client = _EvidenceStageClient(
        content_error=VisionRecognitionError(
            error_code,
            "private provider failure",
            diagnostic={"operation": "content_recognition", "status_code": 503},
        )
    )

    values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path, client
    )

    assert len(values) == 2
    assert all(value["answer_status"] == "unresolved" for value in values)
    assert diagnostic["content_error_diagnostics"] == [
        {
            "mark_ids": [0, 1],
            "error_code": error_code,
            "diagnostic": {
                "operation": "content_recognition",
                "status_code": 503,
            },
        }
    ]
    assert "private provider failure" not in repr(diagnostic)


def test_evidence_mode_partial_content_preserves_completed_and_missing_marks(tmp_path):
    from app.services.vision_recognition import ContentRecognitionResult

    client = _EvidenceStageClient(
        content_result=ContentRecognitionResult(
            items=[_evidence_content_item(0, answer="正确答案")]
        )
    )

    values, localizations, _marks, _diagnostic = _run_evidence_three_stage(
        tmp_path, client
    )

    assert [value["ocr_raw_json"]["evidence_bundle"]["identity"]["mark_id"] for value in values] == [0, 1]
    assert values[0]["answer_status"] == "suggested"
    assert values[0]["ocr_answer"] == "正确答案"
    assert values[1]["answer_status"] == "unresolved"
    assert set(localizations) == {0, 1}


def test_evidence_mode_identical_content_keeps_distinct_mark_geometry_identity(tmp_path):
    from app.services.vision_recognition import ContentRecognitionResult

    client = _EvidenceStageClient(
        content_result=ContentRecognitionResult(
            items=[
                _evidence_content_item(0, raw_text="同文", answer="同答"),
                _evidence_content_item(1, raw_text="同文", answer="同答"),
            ]
        )
    )

    values, _localizations, _marks, _diagnostic = _run_evidence_three_stage(
        tmp_path, client, image_id=123
    )

    identities = [value["ocr_raw_json"]["evidence_bundle"]["identity"] for value in values]
    assert [identity["mark_id"] for identity in identities] == [0, 1]
    assert all(identity["image_id"] == 123 for identity in identities)
    assert identities[0]["question_geometry"] != identities[1]["question_geometry"]


@pytest.mark.parametrize("ocr_status", ["disabled", "unavailable", "available"])
def test_evidence_mode_empty_or_unavailable_ocr_is_neutral(tmp_path, ocr_status):
    from app.services.local_ocr_verification import OCRPageEvidence
    from app.services.vision_recognition import ContentRecognitionResult

    client = _EvidenceStageClient(
        content_result=ContentRecognitionResult(items=[_evidence_content_item(0)])
    )
    page = OCRPageEvidence(
        status=ocr_status,
        error_code="not_configured" if ocr_status == "unavailable" else None,
    )

    values, _localizations, _marks, _diagnostic = _run_evidence_three_stage(
        tmp_path, client, ocr_page_evidence=page
    )

    assert len(values) == 2
    assert values[0]["question_evidence_status"] == "insufficient"
    assert values[0]["ocr_raw_json"]["ocr_page"]["status"] == ocr_status
    assert values[0]["ocr_raw_json"]["selected_ocr_lines"] == []


def test_non_deepseek_client_cannot_forge_independent_deepseek_evidence(tmp_path):
    from app.services.local_ocr_verification import OCRLine, OCRPageEvidence
    from app.services.vision_recognition import ContentObservation, ContentRecognitionResult

    client = _EvidenceStageClient(
        content_result=ContentRecognitionResult(
            items=[
                _evidence_content_item(
                    0,
                    printed_prompt=ContentObservation(
                        source="deepseek",
                        source_class="printed",
                        text="选择词语",
                        bbox=[0.15, 0.15, 0.85, 0.35],
                        confidence=0.9,
                    ),
                )
            ]
        )
    )
    page = OCRPageEvidence(
        status="available",
        lines=[
            OCRLine(
                text="选择词语",
                confidence=0.95,
                bbox=[0.13, 0.14, 0.35, 0.2],
            )
        ],
    )

    values, _localizations, _marks, _diagnostic = _run_evidence_three_stage(
        tmp_path, client, ocr_page_evidence=page
    )

    bundle = values[0]["ocr_raw_json"]["evidence_bundle"]
    assert bundle["question_evidence_status"] == "supported"
    assert all(
        observation["source"] != "deepseek"
        for field in bundle["fields"].values()
        for observation in field["observations"]
    )
    assert values[0]["ocr_raw_json"]["deepseek_observations_trusted"] is False
    assert values[0]["ocr_raw_json"]["deepseek_content"] is None
    assert values[0]["ocr_raw_json"]["untrusted_content"]["printed_prompt"]["source"] == "deepseek"


def test_deepseek_observations_map_to_page_and_ocr_is_spatially_filtered(tmp_path):
    from app.services.deepseek_vision import DeepSeekVisionClient
    from app.services.local_ocr_verification import OCRLine, OCRPageEvidence
    from app.services.vision_recognition import ContentObservation, ContentRecognitionResult

    class DeepSeekEvidenceClient(_EvidenceStageClient, DeepSeekVisionClient):
        @property
        def locate_marked_question_context(self):
            raise AttributeError

    printed = ContentObservation(
        source="deepseek",
        source_class="printed",
        text="选择词语",
        bbox=[0.15, 0.15, 0.85, 0.35],
        confidence=0.9,
    )
    student = ContentObservation(
        source="deepseek",
        source_class="student_handwriting",
        text="学生填写",
        bbox=[0.36, 0.44, 0.64, 0.68],
        confidence=0.88,
    )
    pinyin = ContentObservation(
        source="deepseek",
        source_class="printed",
        text="yǎn jìng",
        bbox=[0.2, 0.75, 0.8, 0.85],
        confidence=0.87,
    )
    client = DeepSeekEvidenceClient(
        content_result=ContentRecognitionResult(
            items=[
                _evidence_content_item(
                    0,
                    answer="正确答案",
                    printed_prompt=printed,
                    printed_pinyin=pinyin,
                    student_handwriting=student,
                )
            ]
        )
    )
    page = OCRPageEvidence(
        status="available",
        lines=[
            OCRLine(
                text="选择词语",
                confidence=0.95,
                bbox=[0.13, 0.14, 0.35, 0.2],
            ),
            OCRLine(
                text="相邻题目",
                confidence=0.99,
                bbox=[0.7, 0.1, 0.9, 0.2],
            ),
            OCRLine(
                text="yǎn jìng",
                confidence=0.93,
                bbox=[0.15, 0.39, 0.35, 0.415],
            ),
        ],
    )

    values, localizations, _marks, _diagnostic = _run_evidence_three_stage(
        tmp_path, client, ocr_page_evidence=page
    )

    first = values[0]
    bundle = first["ocr_raw_json"]["evidence_bundle"]
    prompt_observations = bundle["fields"]["printed_prompt"]["observations"]
    assert bundle["fields"]["printed_prompt"]["status"] == "confirmed"
    assert bundle["fields"]["printed_pinyin"]["status"] == "supported"
    assert bundle["question_evidence_status"] == "supported"
    assert first["ocr_text"] == "学生填写"
    assert first["ocr_answer"] == "正确答案"
    assert first["ocr_raw_json"]["selected_ocr_lines"] == [
        {"text": "选择词语", "confidence": 0.95, "bbox": [0.13, 0.14, 0.35, 0.2]},
        {"text": "yǎn jìng", "confidence": 0.93, "bbox": [0.15, 0.39, 0.35, 0.415]},
    ]
    deepseek_prompt = next(
        observation for observation in prompt_observations if observation["source"] == "deepseek"
    )
    assert deepseek_prompt["bbox"] == pytest.approx(
        [0.124375, 0.1268333333, 0.378125, 0.2115]
    )
    geometry = bundle["identity"]["question_geometry"]
    assert geometry["answer_bbox"] == [0.2, 0.25, 0.3, 0.35]
    assert geometry["prompt_bbox"] == [0.12, 0.12, 0.38, 0.22]
    assert geometry["mark"]["mark_id"] == 0
    assert localizations[0].answer_bbox == [0.2, 0.25, 0.3, 0.35]
    assert "printed_pinyin_bbox" not in first["ocr_raw_json"]["structure_observation"]


def test_production_explicit_printed_role_in_answer_area_becomes_role_conflict(tmp_path):
    from app.services.deepseek_vision import DeepSeekVisionClient
    from app.services.vision_recognition import (
        ContentObservation,
        ContentRecognitionResult,
    )

    class DeepSeekEvidenceClient(_EvidenceStageClient, DeepSeekVisionClient):
        @property
        def locate_marked_question_context(self):
            raise AttributeError

    client = DeepSeekEvidenceClient(
        content_result=ContentRecognitionResult(
            items=[
                _evidence_content_item(
                    0,
                    printed_prompt=ContentObservation(
                        source="deepseek",
                        source_class="printed",
                        text="学生写的字",
                        bbox=[0.36, 0.44, 0.64, 0.68],
                        confidence=0.9,
                    ),
                )
            ]
        )
    )

    values, _localizations, _marks, _diagnostic = _run_evidence_three_stage(
        tmp_path, client
    )

    bundle = values[0]["ocr_raw_json"]["evidence_bundle"]
    assert bundle["fields"]["printed_prompt"]["selected_value"] is None
    assert bundle["fields"]["unknown"]["observations"][0]["text"] == "学生写的字"
    assert bundle["role_conflicts"][0]["claimed_role"] == "printed_prompt"
    assert bundle["question_evidence_status"] == "conflict"


def test_real_deepseek_transport_cannot_self_confirm_pinyin_without_structure(tmp_path):
    import json

    import httpx

    from app.services.deepseek_vision import DeepSeekVisionClient
    from app.services.local_ocr_verification import OCRLine, OCRPageEvidence

    responses = iter(
        [
            {
                "error_marks": [
                    {
                        "mark_id": 0,
                        "mark_type": "cross",
                        "bbox": [0.2, 0.23, 0.3, 0.34],
                        "cross_bbox": None,
                        "circle_bbox": None,
                        "confidence": 0.96,
                    }
                ]
            },
            {
                "items": [
                    {
                        "mark_id": 0,
                        "matched": True,
                        "bbox": [0.1, 0.1, 0.4, 0.45],
                        "answer_bbox": [0.2, 0.25, 0.3, 0.35],
                        "prompt_bbox": [0.12, 0.12, 0.38, 0.22],
                        "question_bbox": None,
                        "incomplete_reason": None,
                        "confidence": 0.94,
                    }
                ]
            },
            {
                "items": [
                    {
                        "mark_id": 0,
                        "raw_text": "学生填写",
                        "instruction": "看拼音写词语",
                        "prompt_text": "题目提示",
                        "normalized_text": None,
                        "answer": None,
                        "subject": "chinese",
                        "question_type": "write_word",
                        "tags": ["词语"],
                        "difficulty": 2,
                        "confidence": 0.9,
                        "uncertain_segments": [],
                        "printed_instruction": None,
                        "printed_prompt": None,
                        "printed_pinyin": {
                            "source": "deepseek",
                            "source_class": "printed",
                            "text": "yǎn jìng",
                            "bbox": [0.2, 0.75, 0.8, 0.85],
                            "confidence": 0.87,
                        },
                        "printed_hanzi": None,
                        "student_handwriting": None,
                        "teacher_correction": None,
                        "uncertain_observations": [],
                    }
                ]
            },
        ]
    )

    def handler(_request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(next(responses), ensure_ascii=False)
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    client = DeepSeekVisionClient(
        api_key="test-key",
        api_host="https://deepseek.invalid/v1",
        timeout_seconds=1,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
        model="deepseek-test",
        thinking="disabled",
        max_tokens=1024,
        image_detail="original",
    )
    page = OCRPageEvidence(
        status="available",
        lines=[
            OCRLine(
                text="yǎn jìng",
                confidence=0.93,
                bbox=[0.15, 0.39, 0.35, 0.415],
            )
        ],
    )

    values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path, client, ocr_page_evidence=page
    )

    bundle = values[0]["ocr_raw_json"]["evidence_bundle"]
    assert diagnostic["content_llm_attempts"] == 1
    assert values[0]["ocr_raw_json"]["deepseek_observations_trusted"] is True
    assert bundle["fields"]["printed_pinyin"]["status"] == "supported"
    assert bundle["question_evidence_status"] == "supported"
    assert "printed_pinyin_bbox" not in values[0]["ocr_raw_json"]["structure_observation"]


def test_real_deepseek_routing_hints_split_ocr_inside_broad_prompt_structure(tmp_path):
    from app.services.local_ocr_verification import OCRLine, OCRPageEvidence

    client = _real_deepseek_transport_client(
        localization_item={
            "mark_id": 0,
            "matched": True,
            "bbox": [0.1, 0.1, 0.45, 0.5],
            "answer_bbox": None,
            "prompt_bbox": [0.11, 0.11, 0.44, 0.47],
            "question_bbox": None,
            "incomplete_reason": None,
            "confidence": 0.94,
        },
        content_item={
            "mark_id": 0,
            "raw_text": "学生填写",
            "instruction": "看拼音写词语",
            "prompt_text": "题目提示",
            "normalized_text": None,
            "answer": None,
            "subject": "chinese",
            "question_type": "write_word",
            "tags": ["词语"],
            "difficulty": 2,
            "confidence": 0.9,
            "uncertain_segments": [],
            "printed_instruction": {
                "source": "deepseek",
                "source_class": "printed",
                "text": "看拼音写词语",
                "bbox": [0.1, 0.1, 0.9, 0.25],
                "confidence": 0.9,
            },
            "printed_prompt": None,
            "printed_pinyin": {
                "source": "deepseek",
                "source_class": "printed",
                "text": "yǎn jìng",
                "bbox": [0.2, 0.65, 0.8, 0.8],
                "confidence": 0.87,
            },
            "printed_hanzi": None,
            "student_handwriting": None,
            "teacher_correction": None,
            "uncertain_observations": [],
        },
    )
    page = OCRPageEvidence(
        status="available",
        lines=[
            OCRLine(
                text="看拼音写词语",
                confidence=0.95,
                bbox=[0.11, 0.11, 0.44, 0.18],
            ),
            OCRLine(
                text="yǎn jìng",
                confidence=0.93,
                bbox=[0.15, 0.375, 0.4, 0.44],
            ),
        ],
    )

    values, _localizations, _marks, _diagnostic = _run_evidence_three_stage(
        tmp_path, client, ocr_page_evidence=page
    )

    raw = values[0]["ocr_raw_json"]
    bundle = raw["evidence_bundle"]
    assert bundle["fields"]["printed_instruction"]["status"] == "supported"
    assert bundle["fields"]["printed_pinyin"]["status"] == "supported"
    assert bundle["fields"]["printed_prompt"]["status"] == "insufficient"
    assert bundle["question_evidence_status"] == "supported"
    for field_name in ("printed_instruction", "printed_pinyin"):
        ocr_observation = next(
            observation
            for observation in bundle["fields"][field_name]["observations"]
            if observation["source"] == "ocr"
        )
        assert ocr_observation["source_class"] == "unknown"
        assert ocr_observation["role"] == "unknown"
    assert raw["structure_observation"] == {
        "printed_prompt_bbox": [0.11, 0.11, 0.44, 0.47]
    }
    assert set(raw["role_routing_hints"]) == {
        "printed_instruction",
        "printed_pinyin",
    }


def test_real_deepseek_routing_hint_exposes_pinyin_conflict_without_structure(tmp_path):
    from app.services.local_ocr_verification import OCRLine, OCRPageEvidence

    client = _real_deepseek_transport_client(
        localization_item={
            "mark_id": 0,
            "matched": True,
            "bbox": [0.1, 0.1, 0.45, 0.5],
            "answer_bbox": None,
            "prompt_bbox": None,
            "question_bbox": None,
            "incomplete_reason": None,
            "confidence": 0.94,
        },
        content_item={
            "mark_id": 0,
            "raw_text": "学生填写",
            "instruction": "看拼音写词语",
            "prompt_text": "题目提示",
            "normalized_text": None,
            "answer": None,
            "subject": "chinese",
            "question_type": "write_word",
            "tags": ["词语"],
            "difficulty": 2,
            "confidence": 0.9,
            "uncertain_segments": [],
            "printed_instruction": None,
            "printed_prompt": None,
            "printed_pinyin": {
                "source": "deepseek",
                "source_class": "printed",
                "text": "pinyin-A",
                "bbox": [0.2, 0.65, 0.8, 0.8],
                "confidence": 0.87,
            },
            "printed_hanzi": None,
            "student_handwriting": None,
            "teacher_correction": None,
            "uncertain_observations": [],
        },
    )
    page = OCRPageEvidence(
        status="available",
        lines=[
            OCRLine(
                text="pinyin-B",
                confidence=0.93,
                bbox=[0.15, 0.375, 0.4, 0.44],
            )
        ],
    )

    values, _localizations, _marks, _diagnostic = _run_evidence_three_stage(
        tmp_path, client, ocr_page_evidence=page
    )

    raw = values[0]["ocr_raw_json"]
    bundle = raw["evidence_bundle"]
    assert raw["structure_observation"] == {}
    assert set(raw["role_routing_hints"]) == {"printed_pinyin"}
    assert bundle["fields"]["printed_pinyin"]["status"] == "conflict"
    assert bundle["fields"]["printed_pinyin"]["conflicts"] == [
        "pinyin-A",
        "pinyin-B",
    ]
    assert bundle["fields"]["unknown"]["observations"] == []
    assert bundle["question_evidence_status"] == "conflict"


def test_plain_timeout_in_evidence_mode_becomes_safe_placeholders(tmp_path):
    client = _EvidenceStageClient(content_error=TimeoutError("private timeout"))

    values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path, client
    )

    assert len(values) == 2
    assert all(value["answer_status"] == "unresolved" for value in values)
    assert diagnostic["content_error_diagnostics"] == [
        {
            "mark_ids": [0, 1],
            "error_code": "vision_timeout",
            "diagnostic": {},
        }
    ]
    assert "private timeout" not in repr(diagnostic)


@pytest.mark.parametrize(
    "localization_error",
    [
        VisionRecognitionError(
            "vision_localization_invalid",
            "定位失败",
            {"operation": "localization"},
        ),
        TimeoutError("private localization timeout"),
    ],
)
def test_evidence_localization_failure_keeps_placeholder_for_every_mark(
    tmp_path, localization_error
):
    from app.services.vision_recognition import ContentRecognitionResult

    class FailingLocalizationClient(_EvidenceStageClient):
        def locate_marked_questions(self, image_path, error_marks, correction=None):
            raise localization_error

    client = FailingLocalizationClient(
        content_result=ContentRecognitionResult(items=[])
    )

    values, localizations, marks, diagnostic = _run_evidence_three_stage(
        tmp_path,
        client,
        localization_stage_retry_count=1,
    )

    assert len(values) == len(localizations) == len(marks) == 2
    assert [
        value["ocr_raw_json"]["evidence_bundle"]["identity"]["mark_id"]
        for value in values
    ] == [0, 1]
    assert all(value["answer_status"] == "unresolved" for value in values)
    assert all(
        localization.bbox_source == "mark_neighborhood_fallback"
        for localization in localizations.values()
    )
    assert diagnostic["localization_error_diagnostics"]


def test_evidence_partial_localization_keeps_success_and_missing_sibling(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionResult,
        MarkQuestionLocalizationItem,
        MarkQuestionLocalizationResult,
    )

    class PartialLocalizationClient(_EvidenceStageClient):
        def locate_marked_questions(self, image_path, error_marks, correction=None):
            return MarkQuestionLocalizationResult(
                items=[
                    MarkQuestionLocalizationItem(
                        mark_id=0,
                        matched=True,
                        bbox=[0.1, 0.1, 0.4, 0.45],
                        answer_bbox=[0.2, 0.25, 0.3, 0.35],
                        prompt_bbox=[0.12, 0.12, 0.38, 0.22],
                        confidence=0.94,
                    )
                ]
            )

    client = PartialLocalizationClient(
        content_result=ContentRecognitionResult(
            items=[_evidence_content_item(0, answer="成功答案")]
        )
    )

    values, localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path,
        client,
        localization_stage_retry_count=0,
        content_batch_size=1,
    )

    assert [value["answer_status"] for value in values] == ["suggested", "unresolved"]
    assert localizations[0].bbox_source == "legacy_mark_localization"
    assert localizations[1].bbox_source == "mark_neighborhood_fallback"
    assert diagnostic["unlocalized_mark_ids"] == []


def test_evidence_empty_localization_result_keeps_all_mark_neighborhoods(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionResult,
        MarkQuestionLocalizationResult,
    )

    class EmptyLocalizationClient(_EvidenceStageClient):
        def locate_marked_questions(self, image_path, error_marks, correction=None):
            return MarkQuestionLocalizationResult(items=[])

    values, localizations, marks, _diagnostic = _run_evidence_three_stage(
        tmp_path,
        EmptyLocalizationClient(content_result=ContentRecognitionResult(items=[])),
    )

    assert len(values) == len(localizations) == len(marks) == 2
    assert all(
        localization.localization_status == "needs_review"
        for localization in localizations.values()
    )


def test_evidence_deadline_after_mark_detection_skips_all_new_calls_and_keeps_marks(
    tmp_path, monkeypatch
):
    from app.services import vision_recognition
    from app.services.vision_recognition import ContentRecognitionResult

    clock = {"now": 9.0}

    class DeadlineClient(_EvidenceStageClient):
        def __init__(self):
            super().__init__(content_result=ContentRecognitionResult(items=[]))
            self.detect_calls = 0
            self.localization_calls = 0

        def detect_marks(self, image_path, local_red_regions, correction=None):
            self.detect_calls += 1
            result = super().detect_marks(image_path, local_red_regions, correction)
            clock["now"] = 11.0
            return result

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            self.localization_calls += 1
            raise AssertionError("expired deadline must prevent localization call")

    client = DeadlineClient()
    monkeypatch.setattr(
        vision_recognition.time,
        "monotonic",
        lambda: clock["now"],
    )

    values, localizations, marks, diagnostic = _run_evidence_three_stage(
        tmp_path,
        client,
        deadline=10.0,
        localization_stage_retry_count=2,
    )

    assert client.detect_calls == 1
    assert client.localization_calls == 0
    assert client.content_calls == 0
    assert len(values) == len(localizations) == len(marks) == 2
    assert diagnostic["deadline_exhausted_stage"] == "mark_localization"


def test_evidence_deadline_blocks_mark_detection_retry_and_keeps_known_mark(
    tmp_path, monkeypatch
):
    from app.services import vision_recognition
    from app.services.error_mark_validation import RedMarkRegion
    from app.services.vision_recognition import (
        ContentRecognitionResult,
        MarkDetectionResult,
    )

    clock = {"now": 9.0}

    class DetectionRetryDeadlineClient(_EvidenceStageClient):
        def __init__(self):
            super().__init__(content_result=ContentRecognitionResult(items=[]))
            self.detect_calls = 0

        def detect_marks(self, image_path, local_red_regions, correction=None):
            self.detect_calls += 1
            clock["now"] = 11.0
            return MarkDetectionResult(error_marks=[_vision_result().error_marks[0]])

    client = DetectionRetryDeadlineClient()
    monkeypatch.setattr(
        vision_recognition.time,
        "monotonic",
        lambda: clock["now"],
    )

    values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path,
        client,
        deadline=10.0,
        mark_stage_retry_count=2,
        local_red_evidence_regions=[
            RedMarkRegion(
                bbox=[0.62, 0.6, 0.73, 0.72],
                pixel_count=100,
                area_ratio=0.0132,
                thinness_ratio=1.1,
            )
        ],
    )

    assert client.detect_calls == 1
    assert len(values) >= 1
    assert values[0]["ocr_raw_json"]["evidence_bundle"]["identity"]["mark_id"] == 0
    assert diagnostic["mark_llm_attempts"] == 1


@pytest.mark.parametrize("retry_entry", ["direct", "targeted"])
@pytest.mark.parametrize("error_kind", ["vision", "timeout"])
def test_evidence_detection_retry_failure_preserves_existing_valid_marks(
    tmp_path, retry_entry, error_kind
):
    from app.services.error_mark_validation import RedMarkRegion
    from app.services.vision_recognition import (
        ContentRecognitionResult,
        MarkDetectionResult,
    )

    def retry_error():
        if error_kind == "vision":
            return VisionRecognitionError(
                "vision_mark_detection_invalid",
                "检测重试失败",
                {"operation": "mark_detection"},
            )
        return TimeoutError("private detection retry timeout")

    class DirectRetryClient(_EvidenceStageClient):
        def __init__(self):
            super().__init__(content_result=ContentRecognitionResult(items=[]))
            self.detect_calls = 0

        def detect_marks(self, image_path, local_red_regions, correction=None):
            self.detect_calls += 1
            if self.detect_calls > 1:
                raise retry_error()
            return MarkDetectionResult(error_marks=[_vision_result().error_marks[0]])

    if retry_entry == "targeted":
        class Client(DirectRetryClient):
            def detect_marks_in_regions(self, image_path, region_ids, correction=None):
                raise retry_error()
    else:
        class Client(DirectRetryClient):
            pass

    client = Client()
    values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path,
        client,
        mark_stage_retry_count=1,
        local_red_evidence_regions=[
            RedMarkRegion(
                bbox=[0.62, 0.6, 0.73, 0.72],
                pixel_count=20,
                area_ratio=0.0132,
                thinness_ratio=1.1,
            )
        ],
    )

    assert any(
        value["ocr_raw_json"]["evidence_bundle"]["identity"]["mark_id"] == 0
        for value in values
    )
    assert diagnostic["mark_error_diagnostics"] == [
        {
            "attempt": 2,
            "entry": retry_entry,
            "error_code": (
                "vision_mark_detection_invalid"
                if error_kind == "vision"
                else "vision_timeout"
            ),
            "diagnostic": (
                {"operation": "mark_detection"}
                if error_kind == "vision"
                else {}
            ),
        }
    ]


@pytest.mark.parametrize("evidence_mode", [True, False])
@pytest.mark.parametrize("error_kind", ["vision", "timeout"])
def test_detection_error_without_prior_valid_mark_still_propagates(
    tmp_path, evidence_mode, error_kind
):
    from app.services.vision_recognition import ContentRecognitionResult

    error = (
        VisionRecognitionError(
            "vision_mark_detection_invalid",
            "检测失败",
            {"operation": "mark_detection"},
        )
        if error_kind == "vision"
        else TimeoutError("initial detection timeout")
    )

    class InitialFailureClient(_EvidenceStageClient):
        def detect_marks(self, image_path, local_red_regions, correction=None):
            raise error

    client = InitialFailureClient(
        content_result=ContentRecognitionResult(items=[])
    )

    with pytest.raises(type(error), match=str(error)):
        _run_evidence_three_stage(
            tmp_path,
            client,
            evidence_mode=evidence_mode,
            image_id="image-initial" if evidence_mode else None,
        )


def test_evidence_deadline_blocks_localization_retry_after_first_empty_result(
    tmp_path, monkeypatch
):
    from app.services import vision_recognition
    from app.services.vision_recognition import (
        ContentRecognitionResult,
        MarkQuestionLocalizationResult,
    )

    clock = {"now": 9.0}

    class RetryDeadlineClient(_EvidenceStageClient):
        def __init__(self):
            super().__init__(content_result=ContentRecognitionResult(items=[]))
            self.localization_calls = 0

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            self.localization_calls += 1
            clock["now"] = 11.0
            return MarkQuestionLocalizationResult(items=[])

    client = RetryDeadlineClient()
    monkeypatch.setattr(
        vision_recognition.time,
        "monotonic",
        lambda: clock["now"],
    )

    values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path,
        client,
        deadline=10.0,
        localization_stage_retry_count=2,
    )

    assert client.localization_calls == 1
    assert len(values) == 2
    assert diagnostic["localization_llm_attempts"] == 1


def test_evidence_deadline_is_checked_before_each_per_mark_localization_call(
    tmp_path, monkeypatch
):
    from app.services import vision_recognition
    from app.services.vision_recognition import (
        ContentRecognitionResult,
        MarkQuestionLocalizationResult,
    )

    clock = {"now": 9.0}

    class PerMarkDeadlineClient(_EvidenceStageClient):
        def __init__(self):
            super().__init__(content_result=ContentRecognitionResult(items=[]))
            self.context_calls = 0
            self.batch_calls = 0

        def locate_marked_question_context(self, image_path, error_mark, correction=None):
            self.context_calls += 1
            clock["now"] = 11.0
            return MarkQuestionLocalizationResult(items=[])

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            self.batch_calls += 1
            raise AssertionError("expired deadline must prevent sibling batch call")

    client = PerMarkDeadlineClient()
    monkeypatch.setattr(
        vision_recognition.time,
        "monotonic",
        lambda: clock["now"],
    )

    values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path,
        client,
        deadline=10.0,
        localization_stage_retry_count=2,
    )

    assert client.context_calls == 1
    assert client.batch_calls == 0
    assert len(values) == 2
    assert diagnostic["deadline_exhausted_stage"] == "mark_localization"


def test_plain_timeout_after_successful_batch_conserves_both_marks(tmp_path):
    from app.services.vision_recognition import ContentRecognitionResult

    class PartialTimeoutClient(_EvidenceStageClient):
        def recognize_localized_content(self, crop_sheet_path, mark_ids, subject_hint):
            self.content_calls += 1
            if mark_ids == [0]:
                return ContentRecognitionResult(
                    items=[_evidence_content_item(0, answer="正确答案")]
                )
            raise TimeoutError("private sibling timeout")

    client = PartialTimeoutClient()

    values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path, client, content_batch_size=1
    )

    assert [value["answer_status"] for value in values] == ["suggested", "unresolved"]
    assert [
        value["ocr_raw_json"]["evidence_bundle"]["identity"]["mark_id"]
        for value in values
    ] == [0, 1]
    assert diagnostic["content_error_diagnostics"][0]["mark_ids"] == [1]


def test_plain_timeout_and_programmer_errors_propagate_in_legacy_mode(tmp_path):
    for error in (TimeoutError("legacy timeout"), AssertionError("programmer bug")):
        client = _EvidenceStageClient(content_error=error)
        with pytest.raises(type(error), match=str(error)):
            _run_evidence_three_stage(
                tmp_path,
                client,
                evidence_mode=False,
                image_id=None,
            )


def test_programmer_error_propagates_in_evidence_mode(tmp_path):
    client = _EvidenceStageClient(content_error=AssertionError("programmer bug"))

    with pytest.raises(AssertionError, match="programmer bug"):
        _run_evidence_three_stage(tmp_path, client)


def test_deadline_exhausted_after_temp_preparation_skips_content_call(
    tmp_path, monkeypatch
):
    import time

    from app.services import vision_recognition
    from app.services.vision_recognition import ContentRecognitionResult

    client = _EvidenceStageClient(content_result=ContentRecognitionResult(items=[]))
    clock = {"now": 9.0}
    monkeypatch.setattr(
        vision_recognition.time,
        "monotonic",
        lambda: clock["now"],
    )
    created_paths = []
    real_named_temporary_file = vision_recognition.tempfile.NamedTemporaryFile

    def tracked_temporary_file(*args, **kwargs):
        temporary = real_named_temporary_file(*args, dir=tmp_path, **kwargs)
        created_paths.append(temporary.name)
        clock["now"] = 11.0
        return temporary

    monkeypatch.setattr(
        vision_recognition.tempfile,
        "NamedTemporaryFile",
        tracked_temporary_file,
    )

    values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
        tmp_path, client, deadline=10.0
    )

    assert client.content_calls == 0
    assert diagnostic["content_llm_attempts"] == 0
    assert diagnostic["content_deadline_exhausted"] is True
    assert len(values) == 2
    assert created_paths
    assert all(not path.exists() for path in map(__import__("pathlib").Path, created_paths))


def test_evidence_mode_expired_deadline_skips_content_and_returns_placeholders(tmp_path):
    from app.services import vision_recognition
    from app.services.vision_recognition import ContentRecognitionResult

    clock = {"now": 9.0}

    class ExpireAfterLocalizationClient(_EvidenceStageClient):
        def locate_marked_questions(self, image_path, error_marks, correction=None):
            result = super().locate_marked_questions(
                image_path, error_marks, correction
            )
            clock["now"] = 11.0
            return result

    client = ExpireAfterLocalizationClient(
        content_result=ContentRecognitionResult(items=[])
    )
    original_monotonic = vision_recognition.time.monotonic
    vision_recognition.time.monotonic = lambda: clock["now"]

    try:
        values, _localizations, _marks, diagnostic = _run_evidence_three_stage(
            tmp_path, client, deadline=10.0
        )
    finally:
        vision_recognition.time.monotonic = original_monotonic

    assert client.content_calls == 0
    assert len(values) == 2
    assert all(value["answer_status"] == "unresolved" for value in values)
    assert diagnostic["content_deadline_exhausted"] is True


def test_evidence_mode_requires_image_identity(tmp_path):
    from app.services.vision_recognition import ContentRecognitionResult

    client = _EvidenceStageClient(content_result=ContentRecognitionResult(items=[]))

    with pytest.raises(ValueError, match="image_id"):
        _run_evidence_three_stage(tmp_path, client, image_id=None)


def test_legacy_false_branch_keeps_all_content_image_failure(tmp_path):
    from app.services.vision_recognition import ContentRecognitionResult, ImageReviewRequired

    client = _EvidenceStageClient(content_result=ContentRecognitionResult(items=[]))

    with pytest.raises(ImageReviewRequired) as raised:
        _run_evidence_three_stage(tmp_path, client, evidence_mode=False, image_id=None)

    assert raised.value.code == "red_marks_unresolved"


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


def test_evidence_batch_returns_task7_values_without_legacy_policy(
    tmp_path, monkeypatch
):
    from app.services import vision_recognition
    from app.services.local_ocr_verification import OCRPageEvidence

    client = object()
    bundle = {
        "schema_version": 1,
        "identity": {
            "image_id": "image-8",
            "mark_id": 4,
            "question_geometry": {"bbox": [0.1, 0.2, 0.3, 0.4]},
        },
    }
    task7_values = [
        {
            "recognition_pipeline": "chinese_marked_evidence_v1",
            "mark_status": "confirmed",
            "question_evidence_status": "supported",
            "answer_status": "suggested",
            "collection_status": "pending_review",
            "review_status": "needs_review",
            "ocr_text": "学生原文",
            "ocr_answer": "建议答案",
            "subject": "chinese",
            "question_type": "write_word",
            "tags": ["词语"],
            "difficulty": 2,
            "crop_region": {"bbox": [0.1, 0.2, 0.3, 0.4]},
            "ocr_raw_json": {"evidence_bundle": bundle},
        }
    ]
    captured = {}

    def fake_three_stage(**kwargs):
        captured.update(kwargs)
        return task7_values, {}, [], {
            "recognition_pipeline": "three_stage",
            "content_llm_ms": 7.5,
            "stage_audit": {},
        }

    class PageOCR:
        enabled = True

        def __init__(self):
            self.page_calls = 0
            self.crop_calls = 0

        def recognize_page(self, image_path, max_edge):
            self.page_calls += 1
            assert max_edge == 987
            return OCRPageEvidence(
                status="available",
                lines=[],
                duration_ms=2.5,
                prepared_size=[400, 300],
            )

        def verify_crop(self, *_args, **_kwargs):
            self.crop_calls += 1
            raise AssertionError("evidence mode must not use legacy crop OCR")

    verifier = PageOCR()
    monkeypatch.setattr(
        vision_recognition,
        "recognize_marked_three_stage",
        fake_three_stage,
    )
    monkeypatch.setattr(
        vision_recognition,
        "build_question_values",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy build path must not run")
        ),
    )
    monkeypatch.setattr(
        vision_recognition,
        "collection_status_for",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy collection policy must not run")
        ),
    )
    monkeypatch.setattr(vision_recognition.time, "monotonic", lambda: 100.0)

    result, values = _run_batch(
        tmp_path,
        client=client,
        ocr_verifier=verifier,
        local_red_scan=_detected_red_scan(),
        evidence_mode=True,
        image_id="image-8",
        deadline=500.0,
        ocr_full_page_max_edge=987,
        ocr_crop_recheck_limit=1,
        evidence_ocr_crop_recheck_limit=9,
        stage_audit_enabled=True,
    )

    assert result.items == []
    assert result.ignored_text == []
    assert result.recognition_pipeline == "chinese_marked_evidence_v1"
    assert values is task7_values
    assert values[0]["ocr_raw_json"]["evidence_bundle"] is bundle
    assert values[0]["ocr_raw_json"]["three_stage"]["content_llm_ms"] == 7.5
    assert values[0]["ocr_raw_json"]["local_ocr_page"]["status"] == "available"
    assert values[0]["ocr_raw_json"]["local_ocr_page"]["crop_recheck_limit"] == 9
    assert values[0]["ocr_raw_json"]["local_ocr_page"]["crop_recheck_count"] == 0
    assert values[0]["ocr_raw_json"]["stage_audit"]["red_scan_ms"] == 1.0
    assert captured["client"] is client
    assert captured["evidence_mode"] is True
    assert captured["image_id"] == "image-8"
    assert captured["deadline"] == 500.0
    assert captured["ocr_page_evidence"].status == "available"
    assert verifier.page_calls == 1
    assert verifier.crop_calls == 0


@pytest.mark.parametrize("page_status", ["disabled", "unavailable"])
def test_evidence_batch_treats_nonavailable_page_ocr_as_neutral(
    tmp_path, monkeypatch, page_status
):
    from app.services import vision_recognition
    from app.services.local_ocr_verification import OCRPageEvidence

    captured = {}

    def fake_three_stage(**kwargs):
        captured.update(kwargs)
        return [], {}, [], {"recognition_pipeline": "three_stage"}

    class NeutralOCR:
        enabled = page_status != "disabled"

        def __init__(self):
            self.page_calls = 0
            self.crop_calls = 0

        def recognize_page(self, _image_path, _max_edge):
            self.page_calls += 1
            return OCRPageEvidence(
                status=page_status,
                error_code=("not_configured" if page_status == "unavailable" else None),
            )

        def verify_crop(self, *_args, **_kwargs):
            self.crop_calls += 1
            raise AssertionError("crop OCR is optional and unused in initial wiring")

    verifier = NeutralOCR()
    monkeypatch.setattr(
        vision_recognition,
        "recognize_marked_three_stage",
        fake_three_stage,
    )

    result, values = _run_batch(
        tmp_path,
        client=object(),
        ocr_verifier=verifier,
        local_red_scan=_detected_red_scan(),
        evidence_mode=True,
        image_id="image-neutral",
        ocr_crop_recheck_limit=1,
    )

    assert result.items == []
    assert values == []
    assert captured["ocr_page_evidence"].status == page_status
    assert verifier.page_calls == 1
    assert verifier.crop_calls == 0


def test_evidence_batch_skips_optional_page_ocr_after_deadline(
    tmp_path, monkeypatch
):
    from app.services import vision_recognition

    captured = {}

    def fake_three_stage(**kwargs):
        captured.update(kwargs)
        return [], {}, [], {"recognition_pipeline": "three_stage"}

    class UnexpectedOCR:
        enabled = True

        def __init__(self):
            self.page_calls = 0

        def recognize_page(self, _image_path, _max_edge):
            self.page_calls += 1
            raise AssertionError("expired optional OCR must be skipped")

    verifier = UnexpectedOCR()
    monkeypatch.setattr(vision_recognition.time, "monotonic", lambda: 11.0)
    monkeypatch.setattr(
        vision_recognition,
        "recognize_marked_three_stage",
        fake_three_stage,
    )

    result, values = _run_batch(
        tmp_path,
        client=object(),
        ocr_verifier=verifier,
        local_red_scan=_detected_red_scan(),
        evidence_mode=True,
        image_id="image-expired",
        deadline=10.0,
    )

    assert result.items == []
    assert values == []
    assert captured["ocr_page_evidence"] is None
    assert captured["deadline"] == 10.0
    assert verifier.page_calls == 0


def test_force_unmarked_never_enters_requested_evidence_branch(
    tmp_path, monkeypatch
):
    from app.services import vision_recognition

    client = FakeClient()
    monkeypatch.setattr(
        vision_recognition,
        "recognize_marked_three_stage",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("force_unmarked must remain legacy")
        ),
    )

    result, values = _run_batch(
        tmp_path,
        client=client,
        local_red_scan=_detected_red_scan(),
        force_mode="unmarked",
        evidence_mode=True,
        image_id="image-force-unmarked",
    )

    assert len(result.items) == 2
    assert len(values) == 2
    assert client.recognize_calls == 1
    assert client.localize_calls == 1


def _pending_task7_value(image_id="image-late", mark_id=0):
    return {
        "recognition_pipeline": "chinese_marked_evidence_v1",
        "mark_status": "needs_review",
        "question_evidence_status": "insufficient",
        "answer_status": "unresolved",
        "collection_status": "pending_review",
        "review_status": "needs_review",
        "ocr_text": None,
        "ocr_answer": None,
        "subject": "chinese",
        "question_type": None,
        "tags": [],
        "difficulty": None,
        "crop_region": {"mark_ids": [mark_id]},
        "ocr_raw_json": {
            "evidence_bundle": {
                "schema_version": 1,
                "identity": {
                    "image_id": image_id,
                    "mark_id": mark_id,
                    "question_geometry": {"bbox": [0.1, 0.2, 0.3, 0.4]},
                },
            }
        },
    }


def test_scan_miss_then_model_mark_promotion_redispatches_to_evidence(
    tmp_path, monkeypatch
):
    from app.services import vision_recognition
    from app.services.error_mark_validation import RedMarkScanResult

    client = FakeClient()
    captured = {}
    task7_values = [_pending_task7_value()]

    def fake_three_stage(**kwargs):
        captured.update(kwargs)
        return task7_values, {}, [], {"recognition_pipeline": "three_stage"}

    monkeypatch.setattr(
        vision_recognition,
        "recognize_marked_three_stage",
        fake_three_stage,
    )
    monkeypatch.setattr(
        vision_recognition,
        "build_question_values",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("late-promoted evidence must not use legacy builder")
        ),
    )
    scan_miss = RedMarkScanResult(
        status="none",
        regions=[],
        red_pixel_count=0,
        scanned_width=400,
        scanned_height=300,
        duration_ms=1.0,
    )

    result, values = _run_batch(
        tmp_path,
        client=client,
        local_red_scan=scan_miss,
        evidence_mode=True,
        image_id="image-late",
    )

    assert client.recognize_calls == 1
    assert client.localize_calls == 0
    assert captured["client"] is client
    assert captured["evidence_mode"] is True
    assert result.items == []
    assert values is task7_values
    assert values[0]["recognition_pipeline"] == "chinese_marked_evidence_v1"
    assert values[0]["collection_status"] == "pending_review"
    assert all(value["collection_status"] != "collected" for value in values)


@pytest.mark.parametrize(
    ("engine_result", "engine_error", "expected_error_code"),
    [
        ({"score": "not-a-number"}, None, "page_ocr_invalid"),
        ({"score": float("nan")}, None, "page_ocr_invalid"),
        (None, TimeoutError("private OCR timeout"), "inference_failed"),
    ],
)
def test_evidence_page_ocr_real_rapidocr_failures_are_neutral(
    tmp_path,
    monkeypatch,
    engine_result,
    engine_error,
    expected_error_code,
):
    from types import SimpleNamespace

    from app.services import vision_recognition
    from app.services.local_ocr_verification import RapidOCRVerifier

    class Engine:
        def __call__(self, *_args, **_kwargs):
            if engine_error is not None:
                raise engine_error
            return SimpleNamespace(
                txts=["题面"],
                scores=[engine_result["score"]],
                boxes=[[[10, 10], [50, 10], [50, 30], [10, 30]]],
            )

    verifier = RapidOCRVerifier(
        enabled=True,
        library_version="test",
        engine_name="test",
        model_version="test",
        model_type="test",
        model_path="test",
        max_pixels=40_000_000,
        line_confidence_threshold=0.85,
        min_effective_characters=2,
        support_similarity_threshold=0.8,
        contradiction_similarity_threshold=0.9,
        engine_factory=Engine,
    )
    captured = {}

    def fake_three_stage(**kwargs):
        captured.update(kwargs)
        return [_pending_task7_value("image-ocr")], {}, [], {
            "recognition_pipeline": "three_stage"
        }

    monkeypatch.setattr(
        vision_recognition,
        "recognize_marked_three_stage",
        fake_three_stage,
    )

    _result, values = _run_batch(
        tmp_path,
        client=object(),
        ocr_verifier=verifier,
        local_red_scan=_detected_red_scan(),
        evidence_mode=True,
        image_id="image-ocr",
    )

    page = captured["ocr_page_evidence"]
    assert page.status == "unavailable"
    assert page.error_code == expected_error_code
    assert "private OCR timeout" not in repr(page.model_dump(mode="json"))
    assert len(values) == 1
    assert values[0]["collection_status"] == "pending_review"


@pytest.mark.parametrize(
    "ocr_error",
    [
        TimeoutError("private built-in timeout"),
        __import__("httpx").ReadTimeout("private HTTP timeout"),
    ],
)
def test_evidence_page_ocr_boundary_timeouts_are_neutral(
    tmp_path, monkeypatch, ocr_error
):
    from app.services import vision_recognition

    class TimeoutOCR:
        enabled = True

        def recognize_page(self, _image_path, _max_edge):
            raise ocr_error

    captured = {}

    def fake_three_stage(**kwargs):
        captured.update(kwargs)
        return [_pending_task7_value("image-timeout")], {}, [], {
            "recognition_pipeline": "three_stage"
        }

    monkeypatch.setattr(
        vision_recognition,
        "recognize_marked_three_stage",
        fake_three_stage,
    )

    _result, values = _run_batch(
        tmp_path,
        client=object(),
        ocr_verifier=TimeoutOCR(),
        local_red_scan=_detected_red_scan(),
        evidence_mode=True,
        image_id="image-timeout",
    )

    assert captured["ocr_page_evidence"].status == "unavailable"
    assert captured["ocr_page_evidence"].error_code == "page_ocr_timeout"
    assert "private" not in repr(captured["ocr_page_evidence"].model_dump(mode="json"))
    assert len(values) == 1


def test_evidence_page_ocr_programmer_error_propagates(tmp_path, monkeypatch):
    from app.services import vision_recognition

    class BrokenOCR:
        enabled = True

        def recognize_page(self, _image_path, _max_edge):
            raise AssertionError("programmer bug")

    monkeypatch.setattr(
        vision_recognition,
        "recognize_marked_three_stage",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("three-stage must not run after programmer error")
        ),
    )

    with pytest.raises(AssertionError, match="programmer bug"):
        _run_batch(
            tmp_path,
            client=object(),
            ocr_verifier=BrokenOCR(),
            local_red_scan=_detected_red_scan(),
            evidence_mode=True,
            image_id="image-programmer-error",
        )


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


def test_actual_evidence_marked_uses_dedicated_localization_recheck_limit(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionResult,
        MarkQuestionLocalizationResult,
    )

    class CountingEmptyLocalizationClient(_EvidenceStageClient):
        def __init__(self):
            super().__init__(content_result=ContentRecognitionResult(items=[]))
            self.localization_calls = 0

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            self.localization_calls += 1
            return MarkQuestionLocalizationResult(items=[])

    client = CountingEmptyLocalizationClient()

    _result, values = _run_batch(
        tmp_path,
        client=client,
        local_red_scan=_detected_red_scan(),
        evidence_mode=True,
        image_id="image-recheck",
        localization_stage_retry_count=0,
        evidence_localization_recheck_limit=2,
    )

    assert client.localization_calls == 3
    assert len(values) == 2


def test_flag_off_marked_ignores_evidence_localization_limit(tmp_path):
    from app.services.vision_recognition import (
        ContentRecognitionResult,
        ImageReviewRequired,
        MarkQuestionLocalizationResult,
    )

    class CountingLegacyLocalizationClient(_EvidenceStageClient):
        def __init__(self):
            super().__init__(content_result=ContentRecognitionResult(items=[]))
            self.localization_calls = 0

        def locate_marked_questions(self, image_path, error_marks, correction=None):
            self.localization_calls += 1
            return MarkQuestionLocalizationResult(items=[])

    client = CountingLegacyLocalizationClient()

    with pytest.raises(ImageReviewRequired):
        _run_batch(
            tmp_path,
            client=client,
            local_red_scan=_detected_red_scan(),
            three_stage_enabled=True,
            evidence_mode=False,
            localization_stage_retry_count=0,
            evidence_localization_recheck_limit=2,
        )

    assert client.localization_calls == 1


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
