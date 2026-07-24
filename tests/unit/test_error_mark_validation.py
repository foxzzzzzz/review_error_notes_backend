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

    valid, rejected = filter_valid_error_marks(
        str(image_path),
        [_mark()],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.05,
    )

    assert [mark.mark_id for mark in valid] == [0]
    assert rejected == []


def test_rejects_white_region_and_low_confidence_mark(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "white.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    valid, rejected = filter_valid_error_marks(
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

    valid, rejected = filter_valid_error_marks(
        str(image_path),
        [_mark([0.0, 0.0, 0.2, 0.2])],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.5,
    )

    assert [mark.mark_id for mark in valid] == [0]
    assert rejected == []


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
