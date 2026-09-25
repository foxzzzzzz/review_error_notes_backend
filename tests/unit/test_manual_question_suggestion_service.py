from types import SimpleNamespace

import pytest

from app.services import manual_question_suggestion as suggestion


def test_ocr_mode_uses_local_text_for_prompt_draft(monkeypatch):
    captured = {}

    class FakeOCR:
        def recognize_crop(self, image_path, bbox):
            captured.update(image_path=image_path, bbox=bbox)
            return SimpleNamespace(
                status="available",
                lines=[
                    SimpleNamespace(text="看拼音写词语"),
                    SimpleNamespace(text="shuǐ guǒ"),
                ],
            )

    monkeypatch.setattr(suggestion, "_new_ocr_verifier", lambda: FakeOCR())

    result = suggestion.recognize_manual_suggestion(
        "/tmp/page.jpg",
        [0.1, 0.2, 0.8, 0.6],
        "ocr",
        subject="chinese",
        grade=1,
        semester=1,
    )

    assert captured == {"image_path": "/tmp/page.jpg", "bbox": [0.1, 0.2, 0.8, 0.6]}
    assert result == {
        "mode": "ocr",
        "fields": {
            "instruction": "",
            "prompt_text": "看拼音写词语\nshuǐ guǒ",
            "question_type": "",
            "correct_answer": "",
            "student_answer": "",
        },
        "ocr_text": "看拼音写词语\nshuǐ guǒ",
    }


def test_llm_mode_uses_context_crop_and_strict_response_schema(monkeypatch):
    captured = {}

    class FakeClient:
        def _request(self, payload, model, diagnostic):
            captured.update(payload=payload, model=model, diagnostic=diagnostic)
            return model.model_validate(
                {
                    "instruction": "看拼音写词语",
                    "prompt_text": "shuǐ guǒ",
                    "question_type": "write_word",
                    "correct_answer": "",
                    "student_answer": "水果",
                }
            )

    monkeypatch.setattr(suggestion, "_image_data_url", lambda path, context, bbox: "data:image/jpeg;base64,AA==")
    monkeypatch.setattr(suggestion.DeepSeekVisionClient, "from_settings", lambda: FakeClient())

    result = suggestion.recognize_manual_suggestion(
        "/tmp/page.jpg",
        [0.4, 0.4, 0.6, 0.6],
        "llm",
        subject="chinese",
        grade=1,
        semester=2,
    )

    assert captured["model"] is suggestion.ManualSuggestionFields
    assert captured["diagnostic"] == {"operation": "manual_question_suggestion"}
    assert result["fields"]["correct_answer"] == ""
    assert result["fields"]["question_type"] == "write_word"
    assert result["ocr_text"] == ""
    with pytest.raises(ValueError):
        suggestion.ManualSuggestionFields.model_validate(
            {
                "instruction": "",
                "prompt_text": "",
                "question_type": "essay",
                "correct_answer": "",
                "student_answer": "",
            }
        )


def test_llm_context_crop_marks_the_selected_bbox_without_mutating_original(tmp_path):
    from PIL import Image
    import base64
    from io import BytesIO

    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    data_url = suggestion._image_data_url(
        str(image_path),
        [0.2, 0.2, 0.8, 0.8],
        [0.4, 0.4, 0.6, 0.6],
    )
    decoded = Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1])))

    assert decoded.size == (60, 60)
    assert decoded.getpixel((20, 20))[2] > decoded.getpixel((20, 20))[0]
    assert decoded.getpixel((5, 5)) == (255, 255, 255)


def test_llm_response_schema_rejects_additional_fields():
    with pytest.raises(ValueError):
        suggestion.ManualSuggestionFields.model_validate(
            {
                "instruction": "",
                "prompt_text": "",
                "question_type": "",
                "correct_answer": "",
                "student_answer": "",
                "extra": "ignored?",
            }
        )
