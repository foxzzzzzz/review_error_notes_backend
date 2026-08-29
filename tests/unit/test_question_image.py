from io import BytesIO

import pytest
from PIL import Image

from app.services.question_image import (
    QuestionImageInvalid,
    QuestionImageNotFound,
    load_cropped_rgb_image,
    load_resized_rgb_image,
    render_numbered_question_sheet,
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


def test_mark_context_preserves_page_bbox_and_converts_local_coordinates(tmp_path):
    from app.services.question_image import (
        local_bbox_to_page,
        page_bbox_to_local,
        render_mark_context,
    )

    source = tmp_path / "context.png"
    Image.new("RGB", (1000, 800), "white").save(source)

    context = render_mark_context(
        source,
        [0.4, 0.4, 0.6, 0.6],
        padding_ratio=1.0,
        max_edge=1200,
        jpeg_quality=90,
        max_pixels=2_000_000,
    )

    assert context.page_bbox == pytest.approx([0.2, 0.2, 0.8, 0.8])
    assert local_bbox_to_page(
        [0.25, 0.25, 0.75, 0.75], context.page_bbox
    ) == pytest.approx([0.35, 0.35, 0.65, 0.65])
    assert page_bbox_to_local(
        [0.35, 0.35, 0.65, 0.65], context.page_bbox
    ) == pytest.approx([0.25, 0.25, 0.75, 0.75])
    assert context.image_bytes.startswith(b"\xff\xd8")


def test_mark_context_clamps_at_page_edges(tmp_path):
    from app.services.question_image import render_mark_context

    source = tmp_path / "edge-context.png"
    Image.new("RGB", (100, 100), "white").save(source)

    context = render_mark_context(
        source,
        [0.0, 0.0, 0.1, 0.1],
        padding_ratio=1.0,
        max_edge=1200,
        jpeg_quality=90,
        max_pixels=20_000,
    )

    assert context.page_bbox == pytest.approx([0.0, 0.0, 0.2, 0.2])


def test_mark_context_reports_actual_pixel_aligned_page_bbox(tmp_path):
    from app.services.question_image import render_mark_context

    source = tmp_path / "fractional-context.png"
    Image.new("RGB", (10, 10), "white").save(source)

    context = render_mark_context(
        source,
        [0.26, 0.26, 0.54, 0.54],
        padding_ratio=0.0,
        max_edge=1200,
        jpeg_quality=90,
        max_pixels=20_000,
    )

    assert context.page_bbox == pytest.approx([0.2, 0.2, 0.6, 0.6])


def test_expand_bbox_to_minimum_context_prevents_tiny_mark_crop():
    from app.services.question_image import expand_bbox_to_minimum_context

    expanded = expand_bbox_to_minimum_context(
        [0.48, 0.48, 0.52, 0.52],
        min_width_ratio=0.22,
        min_height_ratio=0.14,
    )

    assert expanded == pytest.approx([0.39, 0.43, 0.61, 0.57])


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


def test_in_memory_page_resize_preserves_aspect_ratio(tmp_path):
    source = tmp_path / "large.jpg"
    _save_image(source, size=(2000, 1000))

    image = load_resized_rgb_image(source, max_edge=1600, max_pixels=40_000_000)

    assert image.size == (1600, 800)


def test_numbered_question_sheet_contains_one_panel_per_mark(tmp_path):
    source = tmp_path / "source.jpg"
    image = Image.new("RGB", (400, 300), "white")
    image.paste((255, 0, 0), (40, 30, 160, 120))
    image.paste((0, 0, 255), (200, 150, 360, 270))
    image.save(source, format="JPEG")

    content = render_numbered_question_sheet(
        source,
        [(3, [0.1, 0.1, 0.4, 0.4]), (7, [0.5, 0.5, 0.9, 0.9])],
        padding_ratio=0.05,
        max_edge=800,
        jpeg_quality=90,
        max_pixels=40_000_000,
    )

    with Image.open(BytesIO(content)) as sheet:
        assert sheet.mode == "RGB"
        assert max(sheet.size) <= 800
        assert sheet.height > 120


def test_numbered_question_sheet_rejects_empty_localizations(tmp_path):
    source = tmp_path / "source.jpg"
    _save_image(source)

    with pytest.raises(QuestionImageInvalid, match="localizations"):
        render_numbered_question_sheet(
            source,
            [],
            padding_ratio=0.05,
            max_edge=800,
            jpeg_quality=90,
            max_pixels=40_000_000,
        )


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
