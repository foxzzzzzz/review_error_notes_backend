from app.services.vision_recognition import (
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
                bbox=[0.48, 0.26, 0.62, 0.36],
            ),
            VisionItem(
                raw_text="hé zuò",
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
                bbox=[0.04, 0.66, 0.25, 0.77],
            ),
        ],
        ignored_text=[],
    )


class FakeClient:
    def __init__(self, localization_error=False):
        self.recognize_calls = 0
        self.localize_calls = 0
        self.localization_error = localization_error

    def recognize(self, image_path, subject_hint=None):
        self.recognize_calls += 1
        return _vision_result()

    def localize(self, image_path, items):
        self.localize_calls += 1
        if self.localization_error:
            raise VisionRecognitionError("localization failed")
        return LocalizationResult(
            items=[
                LocalizationItem(
                    index=0,
                    matched=True,
                    bbox=[0.45, 0.23, 0.66, 0.4],
                    confidence=0.94,
                ),
                LocalizationItem(
                    index=1,
                    matched=True,
                    bbox=[0.02, 0.63, 0.28, 0.8],
                    confidence=0.91,
                ),
            ]
        )


def test_pipeline_recognizes_and_localizes_once_per_image(tmp_path):
    from app.services.vision_recognition import recognize_question_batch

    client = FakeClient()
    result, values = recognize_question_batch(
        client=client,
        image_path="question.jpg",
        subject_hint="chinese",
        confidence_threshold=0.85,
        localization_threshold=0.85,
        localization_min_iou=0.1,
        tag_config_path=_write_tag_config(tmp_path),
    )

    assert result.items[0].prompt_text == "课文"
    assert client.recognize_calls == 1
    assert client.localize_calls == 1
    assert values[0]["tags"] == ["拼音", "老师批改"]
    assert values[1]["tags"] == ["词语", "错别字"]
    assert values[0]["crop_region"]["bbox"] == [0.45, 0.23, 0.66, 0.4]


def test_pipeline_falls_back_without_candidate_bbox_when_localization_fails(tmp_path):
    from app.services.vision_recognition import recognize_question_batch

    client = FakeClient(localization_error=True)
    _result, values = recognize_question_batch(
        client=client,
        image_path="question.jpg",
        subject_hint="chinese",
        confidence_threshold=0.85,
        localization_threshold=0.85,
        localization_min_iou=0.1,
        tag_config_path=_write_tag_config(tmp_path),
    )

    assert client.localize_calls == 1
    assert all("bbox" not in value["crop_region"] for value in values)
    assert all(value["status"] == "needs_review" for value in values)
