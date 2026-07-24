from io import BytesIO

import pytest
from PIL import Image

from app.services.question_image import (
    QuestionImageInvalid,
    QuestionImageNotFound,
    load_cropped_rgb_image,
    render_question_image,
)


def _save_image(path, size=(100, 80)):
    Image.new("RGB", size, "white").save(path, format="JPEG")


def _rendered_size(content):
    with Image.open(BytesIO(content)) as image:
        return image.size


def test_crop_uses_normalized_ltrb_coordinates(tmp_path):
    source = tmp_path / "source.jpg"
    _save_image(source)

    content = render_question_image(
        source,
        {"bbox": [0.25, 0.25, 0.75, 0.75], "bbox_format": "normalized_ltrb"},
        view="crop",
        jpeg_quality=90,
        max_pixels=40_000_000,
    )

    assert _rendered_size(content) == (50, 40)


def test_in_memory_crop_uses_the_same_normalized_coordinates(tmp_path):
    source = tmp_path / "source.jpg"
    _save_image(source)

    crop = load_cropped_rgb_image(
        source,
        [0.25, 0.25, 0.75, 0.75],
        max_pixels=40_000_000,
    )

    assert crop.mode == "RGB"
    assert crop.size == (50, 40)


def test_original_view_returns_the_complete_image(tmp_path):
    source = tmp_path / "source.jpg"
    _save_image(source)

    content = render_question_image(
        source,
        {"bbox": [0.25, 0.25, 0.75, 0.75], "bbox_format": "normalized_ltrb"},
        view="original",
        jpeg_quality=90,
        max_pixels=40_000_000,
    )

    assert _rendered_size(content) == (100, 80)


def test_original_view_applies_exif_orientation(tmp_path):
    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (40, 20), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(source, format="JPEG", exif=exif)

    content = render_question_image(
        source,
        None,
        view="original",
        jpeg_quality=90,
        max_pixels=40_000_000,
    )

    assert _rendered_size(content) == (20, 40)


@pytest.mark.parametrize("crop_region", [
    None,
    {},
    {"bbox": [0.25, 0.25, 0.75, 0.75], "bbox_format": "legacy_xywh"},
    {"bbox": [0.75, 0.25, 0.25, 0.75], "bbox_format": "normalized_ltrb"},
    {"bbox": [0.25, -0.1, 0.75, 0.75], "bbox_format": "normalized_ltrb"},
])
def test_invalid_crop_region_falls_back_to_the_complete_image(tmp_path, crop_region):
    source = tmp_path / "source.jpg"
    _save_image(source)

    content = render_question_image(
        source,
        crop_region,
        view="crop",
        jpeg_quality=90,
        max_pixels=40_000_000,
    )

    assert _rendered_size(content) == (100, 80)


def test_missing_image_raises_stable_error(tmp_path):
    with pytest.raises(QuestionImageNotFound, match="does not exist"):
        render_question_image(
            tmp_path / "missing.jpg",
            None,
            view="original",
            jpeg_quality=90,
            max_pixels=40_000_000,
        )


def test_corrupt_image_raises_stable_error(tmp_path):
    source = tmp_path / "corrupt.jpg"
    source.write_bytes(b"not-an-image")

    with pytest.raises(QuestionImageInvalid, match="invalid"):
        render_question_image(
            source,
            None,
            view="original",
            jpeg_quality=90,
            max_pixels=40_000_000,
        )


def test_oversized_image_raises_stable_error(tmp_path):
    source = tmp_path / "oversized.jpg"
    _save_image(source, size=(10, 10))

    with pytest.raises(QuestionImageInvalid, match="pixel limit"):
        render_question_image(
            source,
            None,
            view="original",
            jpeg_quality=90,
            max_pixels=50,
        )


def test_decompression_bomb_error_is_mapped_to_stable_error(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    _save_image(source)

    def raise_decompression_bomb(_path):
        raise Image.DecompressionBombError("too many pixels")

    monkeypatch.setattr("app.services.question_image.Image.open", raise_decompression_bomb)

    with pytest.raises(QuestionImageInvalid, match="pixel limit"):
        render_question_image(
            source,
            None,
            view="original",
            jpeg_quality=90,
            max_pixels=40_000_000,
        )
