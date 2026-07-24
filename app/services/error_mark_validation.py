"""Validate that model-proposed error marks contain visible red pixels."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Sequence

from PIL import Image, ImageOps

if TYPE_CHECKING:
    from app.services.vision_recognition import ErrorMark


class ErrorMarkImageInvalid(RuntimeError):
    """The source image could not be safely inspected."""


def _expanded_pixel_box(image_size, bbox, expansion_ratio):
    image_width, image_height = image_size
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    left = max(0.0, left - width * expansion_ratio)
    top = max(0.0, top - height * expansion_ratio)
    right = min(1.0, right + width * expansion_ratio)
    bottom = min(1.0, bottom + height * expansion_ratio)
    return (
        max(0, math.floor(left * image_width)),
        max(0, math.floor(top * image_height)),
        min(image_width, math.ceil(right * image_width)),
        min(image_height, math.ceil(bottom * image_height)),
    )


def _is_red_pixel(pixel) -> bool:
    red, green, blue = pixel
    return (
        red >= 120
        and red - green >= 45
        and red - blue >= 45
        and red >= green * 1.35
        and red >= blue * 1.35
    )


def filter_valid_error_marks(
    image_path: str,
    marks: Sequence[ErrorMark],
    confidence_threshold: float,
    red_pixel_min_ratio: float,
    expansion_ratio: float,
) -> tuple[List[ErrorMark], List[int], List[dict]]:
    """Keep valid marks and report the pixel evidence used for each decision."""
    valid = []
    rejected = []
    diagnostics = []
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            for mark in marks:
                pixel_box = _expanded_pixel_box(
                    image.size,
                    mark.bbox,
                    expansion_ratio,
                )
                crop = image.crop(pixel_box)
                pixel_count = crop.width * crop.height
                red_count = sum(
                    1 for pixel in crop.getdata() if _is_red_pixel(pixel)
                )
                red_pixel_ratio = red_count / pixel_count if pixel_count else 0.0
                if mark.confidence < confidence_threshold:
                    accepted = False
                    reason = "low_confidence"
                elif red_pixel_ratio < red_pixel_min_ratio:
                    accepted = False
                    reason = "insufficient_red_pixels"
                else:
                    accepted = True
                    reason = "accepted"
                diagnostics.append(
                    {
                        "mark_id": mark.mark_id,
                        "confidence": mark.confidence,
                        "confidence_threshold": confidence_threshold,
                        "pixel_box": list(pixel_box),
                        "red_pixel_count": red_count,
                        "pixel_count": pixel_count,
                        "red_pixel_ratio": red_pixel_ratio,
                        "red_pixel_min_ratio": red_pixel_min_ratio,
                        "accepted": accepted,
                        "reason": reason,
                    }
                )
                if accepted:
                    valid.append(mark)
                else:
                    rejected.append(mark.mark_id)
    except (Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ErrorMarkImageInvalid("Error mark image is invalid") from exc
    return valid, rejected, diagnostics
