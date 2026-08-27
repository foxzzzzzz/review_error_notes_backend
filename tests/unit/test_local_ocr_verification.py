import tracemalloc

from PIL import Image

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

    assert result.status == "wrong_candidate"
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
    assert result.error_code == "engine_initialization_failed"


def _verifier(*, enabled=True, engine_factory):
    from app.services.local_ocr_verification import RapidOCRVerifier

    return RapidOCRVerifier(
        enabled=enabled,
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
        engine_factory=engine_factory,
    )


def test_disabled_verifier_never_initializes_engine(tmp_path):
    from PIL import Image

    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)

    def should_not_run():
        raise AssertionError("engine must remain disabled")

    verifier = _verifier(enabled=False, engine_factory=should_not_run)

    assert verifier.verify_crop(
        str(image_path), [0, 0, 1, 1], 0, [_item("课文", "kè wén")]
    ).status == "disabled"
    assert verifier.recognize_page(str(image_path), 1600).status == "disabled"


def test_page_ocr_runs_once_and_returns_normalized_spatial_lines(tmp_path):
    from types import SimpleNamespace
    from PIL import Image

    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    calls = []

    class Engine:
        def __call__(self, image, use_cls=False):
            calls.append(image.shape)
            return SimpleNamespace(
                txts=["课文", "算式"],
                scores=[0.98, 0.97],
                boxes=[
                    [[10, 10], [90, 10], [90, 40], [10, 40]],
                    [[110, 60], [190, 60], [190, 90], [110, 90]],
                ],
            )

    verifier = _verifier(engine_factory=lambda: Engine())
    page = verifier.recognize_page(str(image_path), 1600)

    assert page.status == "available"
    assert page.prepared_size == [200, 100]
    assert page.lines[0].bbox == [0.05, 0.1, 0.45, 0.4]
    assert len(calls) == 1


def test_page_evidence_uses_only_lines_intersecting_candidate_bbox():
    from app.services.local_ocr_verification import (
        OCRLine,
        OCRPageEvidence,
        classify_page_evidence,
    )

    page = OCRPageEvidence(
        status="available",
        lines=[
            OCRLine(text="课文", confidence=0.98, bbox=[0.0, 0.0, 0.45, 0.45]),
            OCRLine(text="算式", confidence=0.98, bbox=[0.55, 0.55, 1.0, 1.0]),
        ],
        duration_ms=1,
        prepared_size=[100, 100],
    )

    result = classify_page_evidence(
        page,
        [0.5, 0.5, 1.0, 1.0],
        target_index=0,
        items=[_item("课文", "kè wén"), _item("算式", "suàn shì")],
        line_confidence_threshold=0.85,
        min_effective_characters=2,
        support_similarity_threshold=0.8,
        contradiction_similarity_threshold=0.9,
    )

    assert result.status == "wrong_candidate"
    assert result.matched_index == 1


def test_dense_red_page_scan_does_not_allocate_one_python_object_per_pixel(tmp_path):
    from app.services.error_mark_validation import scan_red_mark_regions

    image_path = tmp_path / "dense-red.png"
    Image.new("RGB", (800, 800), (220, 30, 30)).save(image_path)

    tracemalloc.start()
    try:
        result = scan_red_mark_regions(
            str(image_path),
            max_edge=1600,
            min_component_pixels=12,
            max_component_area_ratio=0.08,
            max_thinness_ratio=18,
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.status == "none"
    assert peak < 20_000_000
