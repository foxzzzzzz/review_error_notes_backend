import pytest
from PIL import Image, ImageDraw

from app.services.vision_recognition import ErrorMark


def _mark(bbox=None, confidence=0.95):
    return ErrorMark(
        mark_id=0,
        mark_type="circle",
        bbox=bbox or [0.15, 0.15, 0.45, 0.45],
        confidence=confidence,
    )


def test_accepts_mark_box_with_red_pixels(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "marks.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 40, 40), outline=(220, 30, 30), width=5)
    image.save(image_path)

    valid, rejected, _diagnostics = filter_valid_error_marks(
        str(image_path),
        [_mark()],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.05,
    )

    assert [mark.mark_id for mark in valid] == [0]
    assert rejected == []


def test_reports_pixel_evidence_and_rejection_reason_for_each_mark(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "diagnostics.png"
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 40, 40), fill=(220, 30, 30))
    image.save(image_path)

    valid, rejected, diagnostics = filter_valid_error_marks(
        str(image_path),
        [
            _mark([0.2, 0.2, 0.4, 0.4]),
            _mark([0.6, 0.6, 0.8, 0.8]).model_copy(update={"mark_id": 1}),
            _mark([0.2, 0.2, 0.4, 0.4], confidence=0.5).model_copy(
                update={"mark_id": 2}
            ),
        ],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.0,
    )

    assert [mark.mark_id for mark in valid] == [0]
    assert rejected == [1, 2]
    assert diagnostics[0] == {
        "mark_id": 0,
        "confidence": 0.95,
        "confidence_threshold": 0.85,
        "pixel_box": [20, 20, 40, 40],
        "red_pixel_count": 400,
        "pixel_count": 400,
        "red_pixel_ratio": 1.0,
        "red_pixel_min_ratio": 0.01,
        "accepted": True,
        "reason": "accepted",
    }
    assert diagnostics[1]["red_pixel_ratio"] == 0.0
    assert diagnostics[1]["reason"] == "insufficient_red_pixels"
    assert diagnostics[2]["red_pixel_ratio"] == 1.0
    assert diagnostics[2]["reason"] == "low_confidence"


def test_rejects_white_region_and_low_confidence_mark(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "white.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    valid, rejected, _diagnostics = filter_valid_error_marks(
        str(image_path),
        [_mark(), _mark(confidence=0.5).model_copy(update={"mark_id": 1})],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.05,
    )

    assert valid == []
    assert rejected == [0, 1]


def test_expanded_mark_at_image_edge_is_clipped(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "edge.png"
    image = Image.new("RGB", (20, 20), "white")
    ImageDraw.Draw(image).rectangle((0, 0, 4, 4), fill=(230, 20, 20))
    image.save(image_path)

    valid, rejected, _diagnostics = filter_valid_error_marks(
        str(image_path),
        [_mark([0.0, 0.0, 0.2, 0.2])],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.5,
    )

    assert [mark.mark_id for mark in valid] == [0]
    assert rejected == []


def test_localization_bbox_with_red_pixels_is_accepted(tmp_path):
    from app.services.error_mark_validation import (
        validate_localization_red_evidence,
    )

    image_path = tmp_path / "localization-red.png"
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 40, 40), fill=(220, 30, 30))
    image.save(image_path)

    diagnostic = validate_localization_red_evidence(
        str(image_path),
        bbox=[0.15, 0.15, 0.45, 0.45],
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.0,
    )

    assert diagnostic["accepted"] is True
    assert diagnostic["reason"] == "accepted"
    assert diagnostic["bbox"] == [0.15, 0.15, 0.45, 0.45]
    assert diagnostic["pixel_box"] == [15, 15, 45, 45]
    assert diagnostic["expansion_ratio"] == 0.0
    assert diagnostic["red_pixel_count"] > 0
    assert diagnostic["red_pixel_ratio"] >= 0.01


def test_localization_bbox_without_red_pixels_is_rejected(tmp_path):
    from app.services.error_mark_validation import (
        validate_localization_red_evidence,
    )

    image_path = tmp_path / "localization-white.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    diagnostic = validate_localization_red_evidence(
        str(image_path),
        bbox=[0.15, 0.15, 0.45, 0.45],
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.05,
    )

    assert diagnostic["accepted"] is False
    assert diagnostic["reason"] == "insufficient_red_pixels"
    assert diagnostic["red_pixel_count"] == 0


def test_invalid_image_raises_safe_error(tmp_path):
    from app.services.error_mark_validation import (
        ErrorMarkImageInvalid,
        filter_valid_error_marks,
    )

    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"not-an-image")

    with pytest.raises(ErrorMarkImageInvalid):
        filter_valid_error_marks(
            str(image_path),
            [_mark()],
            confidence_threshold=0.85,
            red_pixel_min_ratio=0.01,
            expansion_ratio=0.05,
        )


def test_full_page_scan_keeps_red_marks_and_filters_grid_line_and_noise(tmp_path):
    from app.services.error_mark_validation import scan_red_mark_regions

    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    for y in range(30, 280, 40):
        draw.line((10, y, 390, y), fill=(225, 180, 180), width=1)
    draw.ellipse((40, 50, 100, 110), outline=(210, 25, 25), width=5)
    draw.line((180, 60, 230, 110), fill=(220, 20, 20), width=5)
    draw.line((230, 60, 180, 110), fill=(220, 20, 20), width=5)
    draw.line((20, 10, 380, 10), fill=(210, 25, 25), width=1)
    draw.point((350, 250), fill=(220, 20, 20))
    image.save(image_path)

    result = scan_red_mark_regions(
        str(image_path),
        max_edge=1600,
        min_component_pixels=12,
        max_component_area_ratio=0.08,
        max_thinness_ratio=18,
    )

    assert result.status == "detected"
    assert len(result.regions) == 2
    assert result.red_pixel_count > 0
    assert result.scanned_width == 400
    assert result.scanned_height == 300
    assert result.duration_ms >= 0


def test_full_page_scan_reports_no_reliable_red_marks(tmp_path):
    from app.services.error_mark_validation import scan_red_mark_regions

    image_path = tmp_path / "clean.png"
    Image.new("RGB", (120, 80), "white").save(image_path)

    result = scan_red_mark_regions(
        str(image_path), 1600, 12, 0.08, 18
    )

    assert result.status == "none"
    assert result.regions == []
    assert result.red_pixel_count == 0


def test_full_page_scan_rejects_invalid_image(tmp_path):
    from app.services.error_mark_validation import (
        ErrorMarkImageInvalid,
        scan_red_mark_regions,
    )

    image_path = tmp_path / "broken-scan.jpg"
    image_path.write_bytes(b"not-an-image")

    with pytest.raises(ErrorMarkImageInvalid):
        scan_red_mark_regions(str(image_path), 1600, 12, 0.08, 18)
