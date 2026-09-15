import tracemalloc

from PIL import Image
import pytest

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
        def __call__(self, image, use_cls=False, use_det=True, use_rec=True):
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


def test_page_ocr_restores_detection_after_recognition_only_call(tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    import numpy as np

    class StatefulEngine:
        def __init__(self):
            self.use_det = True
            self.use_rec = True

        def __call__(self, image, use_det=None, use_cls=None, use_rec=None):
            if use_det is not None:
                self.use_det = use_det
            if use_rec is not None:
                self.use_rec = use_rec
            return SimpleNamespace(txts=["题面"] if self.use_rec else [], scores=[0.99],
                boxes=[[[10, 10], [50, 10], [50, 30], [10, 30]]] if self.use_det else None)

    path = tmp_path / "next-page.png"
    Image.new("RGB", (100, 100), "white").save(path)
    engine = StatefulEngine()
    engine(np.zeros((10, 10, 3)), use_det=False, use_rec=False)
    page = _verifier(engine_factory=lambda: engine).recognize_page(str(path), 1600)
    assert page.status == "available"
    assert len(page.lines) == 1
    assert page.lines[0].bbox == [0.1, 0.1, 0.5, 0.3]


def test_recognize_crop_maps_polygon_to_page_coordinates_once(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    import app.services.local_ocr_verification as local_ocr

    image_path = tmp_path / "image.jpg"
    crop = Image.new("RGB", (200, 100), "white")
    calls = []

    def load_crop(*args, **kwargs):
        calls.append("crop")
        return crop

    class Engine:
        def __call__(self, image, **kwargs):
            calls.append("engine")
            return SimpleNamespace(
                txts=["课文"],
                scores=[0.98],
                boxes=[[[20, 10], [100, 10], [100, 60], [20, 60]]],
            )

    monkeypatch.setattr(local_ocr, "load_cropped_rgb_image", load_crop)
    page = _verifier(engine_factory=lambda: Engine()).recognize_crop(
        str(image_path), [0.1, 0.2, 0.9, 0.8]
    )

    assert page.status == "available"
    assert page.prepared_size == [200, 100]
    assert page.lines[0].text == "课文"
    assert page.lines[0].confidence == 0.98
    assert page.lines[0].bbox == pytest.approx([0.18, 0.26, 0.5, 0.56])
    assert calls == ["crop", "engine"]


def test_recognize_crop_preserves_line_without_bbox(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    import app.services.local_ocr_verification as local_ocr

    crop = Image.new("RGB", (100, 100), "white")
    monkeypatch.setattr(local_ocr, "load_cropped_rgb_image", lambda *args, **kwargs: crop)
    page = _verifier(
        engine_factory=lambda: lambda image, **kwargs: SimpleNamespace(
            txts=["课文"], scores=[0.98], boxes=[None]
        )
    ).recognize_crop(str(tmp_path / "image.jpg"), [0, 0, 1, 1])

    assert page.status == "available"
    assert page.lines == [local_ocr.OCRLine(text="课文", confidence=0.98, bbox=None)]


def test_recognize_crop_degrades_invalid_polygons_and_clamps_partial_polygon(
    tmp_path, monkeypatch
):
    from math import inf, nan
    from types import SimpleNamespace
    from PIL import Image
    import app.services.local_ocr_verification as local_ocr

    crop = Image.new("RGB", (100, 100), "white")
    monkeypatch.setattr(local_ocr, "load_cropped_rgb_image", lambda *args, **kwargs: crop)

    page = _verifier(
        engine_factory=lambda: lambda image, **kwargs: SimpleNamespace(
            txts=["空框", "非有限", "完全越界", "部分越界"],
            scores=[0.98, 0.97, 0.96, 0.95],
            boxes=[
                [],
                [[nan, 0], [20, 0], [20, 20], [0, 20]],
                [[150, 10], [200, 10], [200, 30], [150, 30]],
                [[-20, -20], [50, -20], [50, 50], [-20, 50]],
            ],
        )
    ).recognize_crop(str(tmp_path / "image.jpg"), [0.2, 0.2, 0.8, 0.8])

    assert [line.text for line in page.lines] == ["空框", "非有限", "完全越界", "部分越界"]
    assert [line.confidence for line in page.lines] == [0.98, 0.97, 0.96, 0.95]
    assert [line.bbox for line in page.lines[:3]] == [None, None, None]
    assert page.lines[3].bbox == pytest.approx([0.2, 0.2, 0.5, 0.5])

    infinite_page = _verifier(
        engine_factory=lambda: lambda image, **kwargs: SimpleNamespace(
            txts=["无穷"],
            scores=[0.94],
            boxes=[[[inf, 0], [20, 0], [20, 20], [0, 20]]],
        )
    ).recognize_crop(str(tmp_path / "image.jpg"), [0, 0, 1, 1])

    assert infinite_page.lines == [
        local_ocr.OCRLine(text="无穷", confidence=0.94, bbox=None)
    ]


def test_verify_crop_classifies_text_when_polygon_is_empty(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    import app.services.local_ocr_verification as local_ocr

    crop = Image.new("RGB", (100, 100), "white")
    monkeypatch.setattr(local_ocr, "load_cropped_rgb_image", lambda *args, **kwargs: crop)
    verifier = _verifier(
        engine_factory=lambda: lambda image, **kwargs: SimpleNamespace(
            txts=["课文"], scores=[0.98], boxes=[[]]
        )
    )

    result = verifier.verify_crop(
        str(tmp_path / "image.jpg"), [0, 0, 1, 1], 0, [_item("课文", "kè wén")]
    )

    assert result.status == "support"
    assert result.text_summary == "课文"


def test_disabled_recognize_crop_skips_crop_and_engine(tmp_path, monkeypatch):
    import app.services.local_ocr_verification as local_ocr

    def should_not_run(*args, **kwargs):
        raise AssertionError("crop preparation must remain disabled")

    monkeypatch.setattr(local_ocr, "load_cropped_rgb_image", should_not_run)
    page = _verifier(
        enabled=False,
        engine_factory=lambda: (_ for _ in ()).throw(
            AssertionError("engine must remain disabled")
        ),
    ).recognize_crop(str(tmp_path / "image.jpg"), [0, 0, 1, 1])

    assert page.status == "disabled"


def test_recognize_crop_reports_image_preparation_failure(tmp_path, monkeypatch):
    import app.services.local_ocr_verification as local_ocr

    def broken_crop(*args, **kwargs):
        raise OSError("bad crop")

    monkeypatch.setattr(local_ocr, "load_cropped_rgb_image", broken_crop)
    page = _verifier(engine_factory=lambda: AssertionError("engine must not run")).recognize_crop(
        str(tmp_path / "image.jpg"), [0, 0, 1, 1]
    )

    assert page.status == "unavailable"
    assert page.error_code == "image_preparation_failed"
    assert page.duration_ms >= 0


def test_recognize_crop_preserves_engine_error_and_prepared_size(tmp_path, monkeypatch):
    from PIL import Image
    import app.services.local_ocr_verification as local_ocr

    crop = Image.new("RGB", (120, 80), "white")
    monkeypatch.setattr(local_ocr, "load_cropped_rgb_image", lambda *args, **kwargs: crop)

    def broken_factory():
        raise RuntimeError("engine unavailable")

    page = _verifier(engine_factory=broken_factory).recognize_crop(
        str(tmp_path / "image.jpg"), [0, 0, 1, 1]
    )

    assert page.status == "unavailable"
    assert page.error_code == "engine_initialization_failed"
    assert page.prepared_size == [120, 80]


def test_recognize_crop_preserves_inference_error_and_prepared_size(tmp_path, monkeypatch):
    from PIL import Image
    import app.services.local_ocr_verification as local_ocr

    crop = Image.new("RGB", (120, 80), "white")
    monkeypatch.setattr(local_ocr, "load_cropped_rgb_image", lambda *args, **kwargs: crop)

    def broken_engine(image, **kwargs):
        raise RuntimeError("inference unavailable")

    page = _verifier(engine_factory=lambda: broken_engine).recognize_crop(
        str(tmp_path / "image.jpg"), [0, 0, 1, 1]
    )

    assert page.status == "unavailable"
    assert page.error_code == "inference_failed"
    assert page.prepared_size == [120, 80]


def test_verify_crop_uses_raw_path_once_and_retains_classification_semantics(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from PIL import Image
    import app.services.local_ocr_verification as local_ocr

    calls = []
    crop = Image.new("RGB", (100, 100), "white")

    def load_crop(*args, **kwargs):
        calls.append("crop")
        return crop

    class Engine:
        def __call__(self, image, **kwargs):
            calls.append("engine")
            return SimpleNamespace(txts=["课文"], scores=[0.98], boxes=[None])

    monkeypatch.setattr(local_ocr, "load_cropped_rgb_image", load_crop)
    verifier = _verifier(engine_factory=lambda: Engine())
    original_recognize_crop = verifier.recognize_crop

    def recognize_crop(image_path, bbox):
        calls.append("raw")
        return original_recognize_crop(image_path, bbox)

    monkeypatch.setattr(verifier, "recognize_crop", recognize_crop)

    assert verifier.verify_crop(
        str(tmp_path / "image.jpg"), [0, 0, 1, 1], 0, [_item("课文", "kè wén")]
    ).status == "support"
    assert verifier.verify_crop(
        str(tmp_path / "image.jpg"),
        [0, 0, 1, 1],
        0,
        [_item("算式", "suàn shì"), _item("课文", "kè wén")],
    ).status == "wrong_candidate"
    assert verifier.verify_crop(
        str(tmp_path / "image.jpg"), [0, 0, 1, 1], 0, [_item("无关", "wu guan")]
    ).status == "inconclusive"
    assert calls == [
        "raw",
        "crop",
        "engine",
        "raw",
        "crop",
        "engine",
        "raw",
        "crop",
        "engine",
    ]


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
