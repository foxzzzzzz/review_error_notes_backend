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
    for start_y in range(height):
        for x in np.flatnonzero(mask[start_y]):
            x = int(x)
            if visited[start_y, x]:
                continue
            stack = [start_y * width + x]
            visited[start_y, x] = True
            component_pixels = 0
            min_x = max_x = x
            min_y = max_y = start_y
            while stack:
                index = stack.pop()
                x = index % width
                current_y = index // width
                component_pixels += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y)
                for neighbor_x, neighbor_y in (
                    (x - 1, current_y),
                    (x + 1, current_y),
                    (x, current_y - 1),
                    (x, current_y + 1),
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


def _bbox_area(bbox: Sequence[float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _bbox_intersection_area(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _bbox_overlap_ratios(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float]:
    intersection = _bbox_intersection_area(first, second)
    union = _bbox_area(first) + _bbox_area(second) - intersection
    smallest = min(_bbox_area(first), _bbox_area(second))
    return (
        intersection / union if union else 0.0,
        intersection / smallest if smallest else 0.0,
    )


def _bbox_distance(first: Sequence[float], second: Sequence[float]) -> float:
    horizontal_gap = max(first[0] - second[2], 0.0, second[0] - first[2])
    vertical_gap = max(first[1] - second[3], 0.0, second[1] - first[3])
    return math.hypot(horizontal_gap, vertical_gap)


def _union_bbox(first: Sequence[float], second: Sequence[float]) -> List[float]:
    return [
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    ]


def normalize_error_mark_groups(
    marks: Sequence[ErrorMark],
    *,
    dedup_iou_threshold: float,
    pair_max_distance_ratio: float,
) -> tuple[List[ErrorMark], dict]:
    """Deduplicate raw marks and uniquely pair neighboring crosses and circles."""
    deduplicated = []
    duplicate_count = 0
    for mark in marks:
        duplicate_index = None
        for index, existing in enumerate(deduplicated):
            if existing.mark_type != mark.mark_type:
                continue
            iou, containment = _bbox_overlap_ratios(existing.bbox, mark.bbox)
            if max(iou, containment) >= dedup_iou_threshold:
                duplicate_index = index
                break
        if duplicate_index is None:
            deduplicated.append(mark)
            continue
        existing = deduplicated[duplicate_index]
        deduplicated[duplicate_index] = existing.model_copy(
            update={
                "bbox": _union_bbox(existing.bbox, mark.bbox),
                "cross_bbox": existing.cross_bbox or mark.cross_bbox,
                "circle_bbox": existing.circle_bbox or mark.circle_bbox,
                "confidence": max(existing.confidence, mark.confidence),
            }
        )
        duplicate_count += 1

    crosses = [mark for mark in deduplicated if mark.mark_type == "cross"]
    circles = [mark for mark in deduplicated if mark.mark_type == "circle"]
    distances = {
        (cross.mark_id, circle.mark_id): _bbox_distance(cross.bbox, circle.bbox)
        for cross in crosses
        for circle in circles
    }
    pair_by_cross = {}
    paired_circle_ids = set()
    for cross in crosses:
        eligible = sorted(
            (
                (distance, circle.mark_id)
                for circle in circles
                if (distance := distances[(cross.mark_id, circle.mark_id)])
                <= pair_max_distance_ratio
            )
        )
        if not eligible or (
            len(eligible) > 1
            and math.isclose(eligible[0][0], eligible[1][0], abs_tol=1e-9)
        ):
            continue
        distance, circle_id = eligible[0]
        reverse = sorted(
            (
                (distances[(candidate.mark_id, circle_id)], candidate.mark_id)
                for candidate in crosses
                if distances[(candidate.mark_id, circle_id)]
                <= pair_max_distance_ratio
            )
        )
        if len(reverse) > 1 and math.isclose(
            reverse[0][0],
            reverse[1][0],
            abs_tol=1e-9,
        ):
            continue
        if reverse[0][1] != cross.mark_id or circle_id in paired_circle_ids:
            continue
        pair_by_cross[cross.mark_id] = circle_id
        paired_circle_ids.add(circle_id)

    by_id = {mark.mark_id: mark for mark in deduplicated}
    paired_cross_ids = set(pair_by_cross)
    normalized = []
    for mark in deduplicated:
        if mark.mark_id in paired_circle_ids:
            continue
        if mark.mark_id in paired_cross_ids:
            circle = by_id[pair_by_cross[mark.mark_id]]
            normalized.append(
                mark.model_copy(
                    update={
                        "mark_type": "cross_circle",
                        "bbox": _union_bbox(mark.bbox, circle.bbox),
                        "cross_bbox": mark.bbox,
                        "circle_bbox": circle.bbox,
                        "confidence": min(mark.confidence, circle.confidence),
                    }
                )
            )
            continue
        normalized.append(mark)

    normalized = [
        mark.model_copy(update={"mark_id": index})
        for index, mark in enumerate(normalized)
    ]
    paired_count = sum(mark.mark_type == "cross_circle" for mark in normalized)
    return normalized, {
        "raw_mark_count": len(marks),
        "correction_group_count": len(normalized),
        "paired_group_count": paired_count,
        "single_mark_group_count": len(normalized) - paired_count,
        "deduplicated_mark_count": duplicate_count,
    }


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
                component_validation = None
                component_accepted = True
                if mark.mark_type == "cross_circle":
                    if mark.cross_bbox is None or mark.circle_bbox is None:
                        component_accepted = False
                        component_validation = {}
                    else:
                        component_validation = {
                            "cross": _red_pixel_evidence(
                                image,
                                mark.cross_bbox,
                                red_pixel_min_ratio=red_pixel_min_ratio,
                                expansion_ratio=expansion_ratio,
                            ),
                            "circle": _red_pixel_evidence(
                                image,
                                mark.circle_bbox,
                                red_pixel_min_ratio=red_pixel_min_ratio,
                                expansion_ratio=expansion_ratio,
                            ),
                        }
                        component_accepted = all(
                            evidence["accepted"]
                            for evidence in component_validation.values()
                        )
                if mark.confidence < confidence_threshold:
                    accepted = False
                    reason = "low_confidence"
                elif not component_accepted:
                    accepted = False
                    reason = "invalid_component_pixels"
                elif mark.mark_type == "cross_circle":
                    accepted = True
                    reason = "accepted"
                else:
                    accepted = diagnostic["accepted"]
                    reason = diagnostic["reason"]
                mark_diagnostic = {
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
                if component_validation is not None:
                    mark_diagnostic["component_validation"] = component_validation
                diagnostics.append(mark_diagnostic)
                if accepted:
                    valid.append(mark)
                else:
                    rejected.append(mark.mark_id)
    except (Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ErrorMarkImageInvalid("Error mark image is invalid") from exc
    return valid, rejected, diagnostics
