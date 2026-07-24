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
    )


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
    assert all(value["status"] == "needs_review" for value in values)


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
    assert all(value["status"] == "needs_review" for value in values)


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
    assert values[0]["status"] == "confirmed"
    assert verifier.calls == [([0.15, 0.15, 0.4, 0.45], 0)]
    assert "bbox" not in values[1]["crop_region"]
    assert values[1]["status"] == "needs_review"


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

    assert values[1]["status"] == "confirmed"
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

    assert values[1]["status"] == "needs_review"
    assert "bbox" not in values[1]["crop_region"]
    diagnostic = values[1]["ocr_raw_json"]["localization_red_validation"]
    assert diagnostic["accepted"] is False
    assert diagnostic["reason"] == "insufficient_red_pixels"


def test_ocr_contradiction_discards_localized_bbox(tmp_path):
    verifier = FakeOCRVerifier(
        {
            0: OCRVerification(
                status="contradict",
                matched_index=1,
                text_summary="算式",
                confidence=0.98,
            )
        }
    )

    _result, values = _run_batch(tmp_path, ocr_verifier=verifier)

    assert "bbox" not in values[0]["crop_region"]
    assert values[0]["status"] == "needs_review"


def test_ocr_inconclusive_keeps_otherwise_valid_bbox(tmp_path):
    _result, values = _run_batch(tmp_path, ocr_verifier=FakeOCRVerifier())

    assert values[0]["crop_region"]["bbox"] == [0.15, 0.15, 0.4, 0.45]
    assert values[0]["status"] == "confirmed"


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
