import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


class QuestionImageError(Exception):
    pass


class QuestionImageNotFound(QuestionImageError):
    pass


class QuestionImageInvalid(QuestionImageError):
    pass


@dataclass(frozen=True)
class MarkContext:
    image_bytes: bytes
    page_bbox: list[float]


def _validated_normalized_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise QuestionImageInvalid("Question crop bbox is invalid")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        raise QuestionImageInvalid("Question crop bbox is invalid")
    if not all(math.isfinite(value) for value in bbox):
        raise QuestionImageInvalid("Question crop bbox is invalid")
    left, top, right, bottom = bbox
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise QuestionImageInvalid("Question crop bbox is invalid")
    return [float(left), float(top), float(right), float(bottom)]


def local_bbox_to_page(local_bbox, context_page_bbox):
    """Convert a normalized context-crop bbox back to normalized page coordinates."""
    local_left, local_top, local_right, local_bottom = _validated_normalized_bbox(local_bbox)
    page_left, page_top, page_right, page_bottom = _validated_normalized_bbox(
        context_page_bbox
    )
    page_width = page_right - page_left
    page_height = page_bottom - page_top
    return [
        page_left + local_left * page_width,
        page_top + local_top * page_height,
        page_left + local_right * page_width,
        page_top + local_bottom * page_height,
    ]


def page_bbox_to_local(page_bbox, context_page_bbox):
    """Convert a normalized page bbox into normalized context-crop coordinates."""
    left, top, right, bottom = _validated_normalized_bbox(page_bbox)
    context_left, context_top, context_right, context_bottom = (
        _validated_normalized_bbox(context_page_bbox)
    )
    context_width = context_right - context_left
    context_height = context_bottom - context_top
    local_bbox = [
        (left - context_left) / context_width,
        (top - context_top) / context_height,
        (right - context_left) / context_width,
        (bottom - context_top) / context_height,
    ]
    return [min(1.0, max(0.0, value)) for value in local_bbox]


def render_mark_context(
    image_path,
    anchor_bbox,
    padding_ratio,
    max_edge,
    jpeg_quality,
    max_pixels,
):
    """Render a padded mark context and retain its exact normalized page extent."""
    left, top, right, bottom = _validated_normalized_bbox(anchor_bbox)
    width = right - left
    height = bottom - top
    page_bbox = [
        max(0.0, left - width * padding_ratio),
        max(0.0, top - height * padding_ratio),
        min(1.0, right + width * padding_ratio),
        min(1.0, bottom + height * padding_ratio),
    ]
    image = _load_rgb_image(image_path, max_pixels)
    crop_box = _pixel_crop_box(
        image.size,
        {"bbox": page_bbox, "bbox_format": "normalized_ltrb"},
    )
    if crop_box is None:
        raise QuestionImageInvalid("Question crop bbox is invalid")
    pixel_left, pixel_top, pixel_right, pixel_bottom = crop_box
    image_width, image_height = image.size
    page_bbox = [
        pixel_left / image_width,
        pixel_top / image_height,
        pixel_right / image_width,
        pixel_bottom / image_height,
    ]
    context_image = image.crop(crop_box)
    if max(context_image.size) > max_edge:
        context_image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    try:
        output = BytesIO()
        context_image.save(output, format="JPEG", quality=jpeg_quality)
        return MarkContext(image_bytes=output.getvalue(), page_bbox=page_bbox)
    except (OSError, ValueError) as exc:
        raise QuestionImageInvalid("Question image is invalid") from exc


def _pixel_crop_box(image_size, crop_region):
    if not isinstance(crop_region, dict):
        return None
    if crop_region.get("bbox_format") != "normalized_ltrb":
        return None

    bbox = crop_region.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        return None
    if not all(math.isfinite(value) for value in bbox):
        return None

    left, top, right, bottom = bbox
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        return None

    width, height = image_size
    pixel_left = max(0, min(width - 1, math.floor(left * width)))
    pixel_top = max(0, min(height - 1, math.floor(top * height)))
    pixel_right = max(pixel_left + 1, min(width, math.ceil(right * width)))
    pixel_bottom = max(pixel_top + 1, min(height, math.ceil(bottom * height)))
    return pixel_left, pixel_top, pixel_right, pixel_bottom


def _load_rgb_image(image_path, max_pixels):
    path = Path(image_path)
    if not path.is_file():
        raise QuestionImageNotFound("Question image does not exist")

    try:
        with Image.open(path) as source:
            if source.width * source.height > max_pixels:
                raise QuestionImageInvalid("Question image exceeds the pixel limit")
            return ImageOps.exif_transpose(source).convert("RGB")
    except Image.DecompressionBombError as exc:
        raise QuestionImageInvalid("Question image exceeds the pixel limit") from exc
    except (OSError, ValueError) as exc:
        raise QuestionImageInvalid("Question image is invalid") from exc


def load_cropped_rgb_image(image_path, bbox, max_pixels):
    """Load the exact normalized crop used by the question image endpoint."""
    image = _load_rgb_image(image_path, max_pixels)
    crop_box = _pixel_crop_box(
        image.size,
        {"bbox": bbox, "bbox_format": "normalized_ltrb"},
    )
    if crop_box is None:
        raise QuestionImageInvalid("Question crop bbox is invalid")
    return image.crop(crop_box)


def load_resized_rgb_image(image_path, max_edge, max_pixels):
    """Load an EXIF-corrected page and bound its longest edge for OCR."""
    image = _load_rgb_image(image_path, max_pixels)
    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image


def render_numbered_question_sheet(
    image_path,
    localizations,
    padding_ratio,
    max_edge,
    jpeg_quality,
    max_pixels,
):
    """Render stable mark-id labels above padded question crops."""
    if not localizations:
        raise QuestionImageInvalid("Question sheet localizations are empty")
    image = _load_rgb_image(image_path, max_pixels)
    panels = []
    for mark_id, bbox in localizations:
        if not isinstance(mark_id, int) or isinstance(mark_id, bool):
            raise QuestionImageInvalid("Question sheet mark id is invalid")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise QuestionImageInvalid("Question sheet localization is invalid")
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top
        padded = [
            max(0.0, left - width * padding_ratio),
            max(0.0, top - height * padding_ratio),
            min(1.0, right + width * padding_ratio),
            min(1.0, bottom + height * padding_ratio),
        ]
        crop_box = _pixel_crop_box(
            image.size,
            {"bbox": padded, "bbox_format": "normalized_ltrb"},
        )
        if crop_box is None:
            raise QuestionImageInvalid("Question sheet localization is invalid")
        crop = image.crop(crop_box)
        label_height = 28
        panel = Image.new("RGB", (crop.width, crop.height + label_height), "white")
        ImageDraw.Draw(panel).text((8, 7), f"mark_id={mark_id}", fill="black")
        panel.paste(crop, (0, label_height))
        panels.append(panel)

    columns = 1 if len(panels) == 1 else 2
    rows = math.ceil(len(panels) / columns)
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    gap = 8
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + (columns - 1) * gap,
            rows * cell_height + (rows - 1) * gap,
        ),
        "white",
    )
    for index, panel in enumerate(panels):
        column = index % columns
        row = index // columns
        sheet.paste(panel, (column * (cell_width + gap), row * (cell_height + gap)))
    if max(sheet.size) > max_edge:
        sheet.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    try:
        output = BytesIO()
        sheet.save(output, format="JPEG", quality=jpeg_quality)
        return output.getvalue()
    except (OSError, ValueError) as exc:
        raise QuestionImageInvalid("Question image is invalid") from exc


def render_question_image(image_path, crop_region, view, jpeg_quality, max_pixels):
    image = _load_rgb_image(image_path, max_pixels)
    if view == "crop":
        crop_box = _pixel_crop_box(image.size, crop_region)
        if crop_box:
            image = image.crop(crop_box)

    try:
        output = BytesIO()
        image.save(output, format="JPEG", quality=jpeg_quality)
        return output.getvalue()
    except (OSError, ValueError) as exc:
        raise QuestionImageInvalid("Question image is invalid") from exc
