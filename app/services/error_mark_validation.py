"""Validate that model-proposed error marks contain visible red pixels."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, List, Literal, Sequence

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field
import numpy as np

if TYPE_CHECKING:
    from app.services.vision_recognition import ErrorMark


class ErrorMarkImageInvalid(RuntimeError):
    """The source image could not be safely inspected."""


class RedMarkRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    bbox: List[float]
    pixel_count: int = Field(ge=1)
    area_ratio: float = Field(gt=0, le=1)
    thinness_ratio: float = Field(ge=1)


class RedMarkScanResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["detected", "none"]
    regions: List[RedMarkRegion]
    red_pixel_count: int = Field(ge=0)
    scanned_width: int = Field(ge=1)
    scanned_height: int = Field(ge=1)
    duration_ms: float = Field(ge=0)


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


def scan_red_mark_regions(
    image_path: str,
    max_edge: int,
    min_component_pixels: int,
    max_component_area_ratio: float,
    max_thinness_ratio: float,
) -> RedMarkScanResult:
    """Detect bounded red connected components without invoking OCR."""
    started = time.perf_counter()
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if max(image.size) > max_edge:
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            width, height = image.size
    except (Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ErrorMarkImageInvalid("Error mark image is invalid") from exc

    pixels = np.asarray(image, dtype=np.int16)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    mask = (
        (red >= 120)
        & (red - green >= 45)
        & (red - blue >= 45)
        & (red >= green * 1.35)
        & (red >= blue * 1.35)
    )
    visited = np.zeros(mask.shape, dtype=bool)
    red_pixel_count = int(mask.sum())
    regions = []
    while True:
        unvisited = mask & ~visited
        if not unvisited.any():
            break
        start_index = int(unvisited.argmax())
        stack = [start_index]
        visited[start_index // width, start_index % width] = True
        component_pixels = 0
        min_x = max_x = start_index % width
        min_y = max_y = start_index // width
        while stack:
            index = stack.pop()
            x = index % width
            y = index // width
            component_pixels += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for neighbor_x, neighbor_y in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if not (0 <= neighbor_x < width and 0 <= neighbor_y < height):
                    continue
                neighbor = neighbor_y * width + neighbor_x
                if mask[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                    visited[neighbor_y, neighbor_x] = True
                    stack.append(neighbor)

        component_width = max_x - min_x + 1
        component_height = max_y - min_y + 1
        area_ratio = component_width * component_height / (width * height)
        thinness_ratio = max(component_width, component_height) / max(
            1, min(component_width, component_height)
        )
        if (
            component_pixels < min_component_pixels
            or area_ratio > max_component_area_ratio
            or thinness_ratio > max_thinness_ratio
        ):
            continue
        regions.append(
            RedMarkRegion(
                bbox=[
                    min_x / width,
                    min_y / height,
                    (max_x + 1) / width,
                    (max_y + 1) / height,
                ],
                pixel_count=component_pixels,
                area_ratio=area_ratio,
                thinness_ratio=thinness_ratio,
            )
        )

    return RedMarkScanResult(
        status="detected" if regions else "none",
        regions=regions,
        red_pixel_count=red_pixel_count,
        scanned_width=width,
        scanned_height=height,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


def _red_pixel_evidence(
    image: Image.Image,
    bbox,
    red_pixel_min_ratio: float,
    expansion_ratio: float,
) -> dict:
    pixel_box = _expanded_pixel_box(image.size, bbox, expansion_ratio)
    crop = image.crop(pixel_box)
    pixel_count = crop.width * crop.height
    red_count = sum(1 for pixel in crop.getdata() if _is_red_pixel(pixel))
    red_pixel_ratio = red_count / pixel_count if pixel_count else 0.0
    accepted = red_pixel_ratio >= red_pixel_min_ratio
    return {
        "pixel_box": list(pixel_box),
        "red_pixel_count": red_count,
        "pixel_count": pixel_count,
        "red_pixel_ratio": red_pixel_ratio,
        "red_pixel_min_ratio": red_pixel_min_ratio,
        "accepted": accepted,
        "reason": "accepted" if accepted else "insufficient_red_pixels",
    }


def validate_localization_red_evidence(
    image_path: str,
    bbox,
    red_pixel_min_ratio: float,
    expansion_ratio: float,
) -> dict:
    """Report whether a trusted localization bbox contains visible red evidence."""
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            diagnostic = _red_pixel_evidence(
                image,
                bbox,
                red_pixel_min_ratio=red_pixel_min_ratio,
                expansion_ratio=expansion_ratio,
            )
            diagnostic.update(
                {
                    "bbox": list(bbox),
                    "expansion_ratio": expansion_ratio,
                }
            )
            return diagnostic
    except (Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ErrorMarkImageInvalid("Error mark image is invalid") from exc


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
                diagnostic = _red_pixel_evidence(
                    image,
                    mark.bbox,
                    red_pixel_min_ratio=red_pixel_min_ratio,
                    expansion_ratio=expansion_ratio,
                )
                if mark.confidence < confidence_threshold:
                    accepted = False
                    reason = "low_confidence"
                else:
                    accepted = diagnostic["accepted"]
                    reason = diagnostic["reason"]
                diagnostics.append(
                    {
                        "mark_id": mark.mark_id,
                        "confidence": mark.confidence,
                        "confidence_threshold": confidence_threshold,
                        "pixel_box": diagnostic["pixel_box"],
                        "red_pixel_count": diagnostic["red_pixel_count"],
                        "pixel_count": diagnostic["pixel_count"],
                        "red_pixel_ratio": diagnostic["red_pixel_ratio"],
                        "red_pixel_min_ratio": diagnostic[
                            "red_pixel_min_ratio"
                        ],
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
