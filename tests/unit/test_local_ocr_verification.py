from app.services.vision_recognition import VisionItem


def _item(prompt_text, raw_text):
    return VisionItem(
        raw_text=raw_text,
        instruction="完成练习",
        prompt_text=prompt_text,
        normalized_text=raw_text,
        answer=raw_text,
        subject="chinese",
        question_type="other",
        tags=["词语"],
        difficulty=2,
        confidence=0.95,
        uncertain_segments=[],
    )


def _classify(lines, target_index=0):
    from app.services.local_ocr_verification import classify_ocr_lines

    return classify_ocr_lines(
        lines=lines,
        target_index=target_index,
        items=[_item("课文", "kè wén"), _item("算式", "suàn shì")],
        line_confidence_threshold=0.85,
        min_effective_characters=2,
        support_similarity_threshold=0.8,
        contradiction_similarity_threshold=0.9,
    )


def test_matching_target_text_supports_crop():
    from app.services.local_ocr_verification import OCRLine

    result = _classify([OCRLine(text="课文", confidence=0.98)])

    assert result.status == "support"
    assert result.matched_index == 0


def test_text_matching_another_item_better_contradicts_crop():
    from app.services.local_ocr_verification import OCRLine

    result = _classify([OCRLine(text="算式", confidence=0.98)])

    assert result.status == "contradict"
    assert result.matched_index == 1


def test_pinyin_tones_and_punctuation_do_not_create_false_contradiction():
    from app.services.local_ocr_verification import OCRLine

    result = _classify([OCRLine(text="ke wen！", confidence=0.98)])

    assert result.status == "support"


def test_empty_low_confidence_short_and_instruction_only_are_inconclusive():
    from app.services.local_ocr_verification import OCRLine

    cases = [
        [],
        [OCRLine(text="课文", confidence=0.4)],
        [OCRLine(text="课", confidence=0.99)],
        [OCRLine(text="完成练习", confidence=0.99)],
    ]

    assert [_classify(lines).status for lines in cases] == [
        "inconclusive",
        "inconclusive",
        "inconclusive",
        "inconclusive",
    ]


def test_verifier_returns_unavailable_when_engine_initialization_fails(tmp_path):
    from app.services.local_ocr_verification import RapidOCRVerifier

    image_path = tmp_path / "image.jpg"
    from PIL import Image

    Image.new("RGB", (100, 100), "white").save(image_path)

    def broken_factory():
        raise RuntimeError("engine unavailable")

    verifier = RapidOCRVerifier(
        enabled=True,
        library_version="3.9.1",
        engine_name="onnxruntime",
        model_version="PP-OCRv5",
        model_type="mobile",
        model_path="models/ppocrv5",
        max_pixels=40_000_000,
        line_confidence_threshold=0.85,
        min_effective_characters=2,
        support_similarity_threshold=0.8,
        contradiction_similarity_threshold=0.9,
        engine_factory=broken_factory,
    )

    result = verifier.verify(
        str(image_path),
        [0.1, 0.1, 0.9, 0.9],
        target_index=0,
        items=[_item("课文", "kè wén")],
    )

    assert result.status == "unavailable"
    assert result.text_summary == ""
