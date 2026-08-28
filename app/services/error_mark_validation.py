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


def _bbox_contains(
    container: Sequence[float],
    candidate: Sequence[float],
    tolerance: float = 0.0,
) -> bool:
    return (
        container[0] - tolerance <= candidate[0]
        and container[1] - tolerance <= candidate[1]
        and container[2] + tolerance >= candidate[2]
        and container[3] + tolerance >= candidate[3]
    )


def group_red_evidence_regions(
    regions: Sequence[RedMarkRegion],
    *,
    max_gap_ratio: float,
    max_group_area_ratio: float,
) -> tuple[List[RedMarkRegion], dict]:
    """Fuse nearby red fragments into bounded anti-miss evidence groups."""
    groups: List[RedMarkRegion] = []
    for region in regions:
        matching_indexes = []
        for index, existing in enumerate(groups):
            union = _union_bbox(existing.bbox, region.bbox)
            union_area = (union[2] - union[0]) * (union[3] - union[1])
            if (
                _bbox_distance(existing.bbox, region.bbox) <= max_gap_ratio
                and union_area <= max_group_area_ratio
            ):
                matching_indexes.append(index)
        if not matching_indexes:
            groups.append(region.model_copy(deep=True))
            continue
        target_index = matching_indexes[0]
        matched_groups = [groups[index] for index in matching_indexes]
        union = list(region.bbox)
        for matched_group in matched_groups:
            union = _union_bbox(union, matched_group.bbox)
        union_area = (union[2] - union[0]) * (union[3] - union[1])
        if union_area > max_group_area_ratio:
            matched_groups = [groups[target_index]]
            matching_indexes = [target_index]
            union = _union_bbox(groups[target_index].bbox, region.bbox)
        width = union[2] - union[0]
        height = union[3] - union[1]
        merged = RedMarkRegion(
            bbox=union,
            pixel_count=region.pixel_count
            + sum(group.pixel_count for group in matched_groups),
            area_ratio=width * height,
            thinness_ratio=max(width, height) / max(min(width, height), 1e-9),
        )
        for index in reversed(matching_indexes):
            groups.pop(index)
        groups.insert(target_index, merged)
    return groups, {
        "raw_component_count": len(regions),
        "evidence_group_count": len(groups),
    }


def merge_error_mark_attempts(
    attempts: Sequence[Sequence["ErrorMark"]],
    *,
    dedup_iou_threshold: float,
) -> tuple[List["ErrorMark"], dict]:
    """Merge validated primitives from every LLM attempt before pairing."""
    merged: List["ErrorMark"] = []
    duplicate_count = 0
    for attempt in attempts:
        for mark in attempt:
            duplicate_index = None
            for index, existing in enumerate(merged):
                if existing.mark_type != mark.mark_type:
                    continue
                iou, containment = _bbox_overlap_ratios(existing.bbox, mark.bbox)
                if max(iou, containment) >= dedup_iou_threshold:
                    duplicate_index = index
                    break
            if duplicate_index is None:
                merged.append(mark)
                continue
            duplicate_count += 1
            if mark.confidence > merged[duplicate_index].confidence:
                merged[duplicate_index] = mark
    merged = [
        mark.model_copy(update={"mark_id": index})
        for index, mark in enumerate(merged)
    ]
    return merged, {
        "attempt_primitive_counts": [len(attempt) for attempt in attempts],
        "merged_primitive_count": len(merged),
        "cross_attempt_deduplicated_count": duplicate_count,
    }


def normalize_error_mark_groups(
    marks: Sequence[ErrorMark],
    *,
    dedup_iou_threshold: float,
    pair_max_distance_ratio: float,
    pair_max_relative_distance_ratio: float = 1.0,
    pair_min_margin_ratio: float = 0.2,
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
    edges = {}
    for cross in crosses:
        for circle in circles:
            distance = _bbox_distance(cross.bbox, circle.bbox)
            circle_scale = max(
                circle.bbox[2] - circle.bbox[0],
                circle.bbox[3] - circle.bbox[1],
                1e-9,
            )
            relative_distance = distance / circle_scale
            edges[(cross.mark_id, circle.mark_id)] = {
                "distance": distance,
                "relative_distance": relative_distance,
                "intersects": distance == 0,
                "eligible": distance <= pair_max_distance_ratio
                and relative_distance <= pair_max_relative_distance_ratio,
            }

    def unique_best(candidates):
        eligible = sorted(
            candidates,
            key=lambda candidate: (
                not candidate[2]["intersects"],
                candidate[2]["distance"],
                candidate[2]["relative_distance"],
                candidate[1],
            ),
        )
        if not eligible:
            return None, "no_eligible_candidate", None
        if len(eligible) == 1:
            return eligible[0], "unique", None
        first, second = eligible[:2]
        first_distance = first[2]["distance"]
        second_distance = second[2]["distance"]
        margin = (
            (second_distance - first_distance) / max(second_distance, 1e-9)
            if second_distance > 0
            else 0.0
        )
        if margin < pair_min_margin_ratio:
            return None, "ambiguous_margin", margin
        return first, "clear_margin", margin

    pair_by_cross = {}
    paired_circle_ids = set()
    pair_diagnostics = []
    for cross in crosses:
        candidates = [
            (cross.mark_id, circle.mark_id, edges[(cross.mark_id, circle.mark_id)])
            for circle in circles
            if edges[(cross.mark_id, circle.mark_id)]["eligible"]
        ]
        best, reason, margin = unique_best(candidates)
        if best is None:
            if not candidates and circles:
                nearest_circle = min(
                    circles,
                    key=lambda circle: edges[(cross.mark_id, circle.mark_id)]["distance"],
                )
                nearest = edges[(cross.mark_id, nearest_circle.mark_id)]
                reason = (
                    "relative_distance_exceeded"
                    if nearest["distance"] <= pair_max_distance_ratio
                    else "page_distance_exceeded"
                )
                circle_id = nearest_circle.mark_id
            else:
                circle_id = candidates[0][1] if candidates else None
                nearest = candidates[0][2] if candidates else None
            pair_diagnostics.append({
                "cross_id": cross.mark_id,
                "circle_id": circle_id,
                "accepted": False,
                "reason": reason,
                "pair_tier": None,
                "distance_ratio": round(nearest["distance"], 6) if nearest else None,
                "relative_distance_ratio": round(nearest["relative_distance"], 6) if nearest else None,
                "margin_ratio": round(margin, 6) if margin is not None else None,
            })
            continue
        _, circle_id, edge = best
        reverse_candidates = [
            (circle_id, candidate.mark_id, edges[(candidate.mark_id, circle_id)])
            for candidate in crosses
            if edges[(candidate.mark_id, circle_id)]["eligible"]
        ]
        reverse, reverse_reason, _reverse_margin = unique_best(reverse_candidates)
        if reverse is None or reverse[1] != cross.mark_id or circle_id in paired_circle_ids:
            pair_diagnostics.append({
                "cross_id": cross.mark_id,
                "circle_id": circle_id,
                "accepted": False,
                "reason": reverse_reason if reverse is None else "not_mutual_best",
                "pair_tier": None,
                "distance_ratio": round(edge["distance"], 6),
                "relative_distance_ratio": round(edge["relative_distance"], 6),
                "margin_ratio": round(margin, 6) if margin is not None else None,
            })
            continue
        pair_by_cross[cross.mark_id] = circle_id
        paired_circle_ids.add(circle_id)
        pair_diagnostics.append({
            "cross_id": cross.mark_id,
            "circle_id": circle_id,
            "accepted": True,
            "reason": "mutual_unique_best",
            "pair_tier": "strong" if edge["intersects"] else "nearby_review",
            "distance_ratio": round(edge["distance"], 6),
            "relative_distance_ratio": round(edge["relative_distance"], 6),
            "margin_ratio": round(margin, 6) if margin is not None else None,
        })

    by_id = {mark.mark_id: mark for mark in deduplicated}
    paired_cross_ids = set(pair_by_cross)
    normalized = []
    nearby_review_cross_ids = {
        item["cross_id"]
        for item in pair_diagnostics
        if item["accepted"] and item["pair_tier"] == "nearby_review"
    }
    review_required_mark_ids = []
    for mark in deduplicated:
        if mark.mark_id in paired_circle_ids:
            continue
        if mark.mark_id in paired_cross_ids:
            circle = by_id[pair_by_cross[mark.mark_id]]
            if mark.mark_id in nearby_review_cross_ids:
                review_required_mark_ids.append(len(normalized))
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
    diagnostic = {
        "raw_mark_count": len(marks),
        "correction_group_count": len(normalized),
        "paired_group_count": paired_count,
        "single_mark_group_count": len(normalized) - paired_count,
        "deduplicated_mark_count": duplicate_count,
        "review_required_mark_ids": review_required_mark_ids,
    }
    if pair_diagnostics:
        diagnostic["pair_diagnostics"] = pair_diagnostics
    return normalized, diagnostic


def filter_valid_error_marks(
    image_path: str,
    marks: Sequence[ErrorMark],
    confidence_threshold: float,
    red_pixel_min_ratio: float,
    expansion_ratio: float,
    component_fallback_enabled: bool = False,
    component_pair_max_distance_ratio: float = 1.0,
) -> tuple[List[ErrorMark], List[int], List[dict]]:
    """Keep valid marks and report the pixel evidence used for each decision."""
    valid = []
    rejected = []
    diagnostics = []
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            next_fallback_mark_id = max(
                (mark.mark_id for mark in marks),
                default=-1,
            ) + 1
            for mark in marks:
                diagnostic = _red_pixel_evidence(
                    image,
                    mark.bbox,
                    red_pixel_min_ratio=red_pixel_min_ratio,
                    expansion_ratio=expansion_ratio,
                )
                component_validation = None
                component_accepted = True
                components_within_union = None
                component_pair_distance_ratio = None
                components_pairable = None
                fallback_marks = []
                fallback_type = None
                if mark.mark_type == "cross_circle":
                    component_validation = {}
                    if mark.cross_bbox is not None:
                        component_validation["cross"] = _red_pixel_evidence(
                            image,
                            mark.cross_bbox,
                            red_pixel_min_ratio=red_pixel_min_ratio,
                            expansion_ratio=expansion_ratio,
                        )
                    if mark.circle_bbox is not None:
                        component_validation["circle"] = _red_pixel_evidence(
                            image,
                            mark.circle_bbox,
                            red_pixel_min_ratio=red_pixel_min_ratio,
                            expansion_ratio=expansion_ratio,
                        )
                    component_accepted = (
                        set(component_validation) == {"cross", "circle"}
                        and all(
                            evidence["accepted"]
                            for evidence in component_validation.values()
                        )
                    )
                    if (
                        mark.cross_bbox is not None
                        and mark.circle_bbox is not None
                    ):
                        components_within_union = _bbox_contains(
                            mark.bbox,
                            mark.cross_bbox,
                            expansion_ratio,
                        ) and _bbox_contains(
                            mark.bbox,
                            mark.circle_bbox,
                            expansion_ratio,
                        )
                        component_pair_distance_ratio = _bbox_distance(
                            mark.cross_bbox,
                            mark.circle_bbox,
                        )
                        components_pairable = (
                            component_pair_distance_ratio
                            <= component_pair_max_distance_ratio
                        )
                    if (
                        component_fallback_enabled
                        and mark.confidence >= confidence_threshold
                    ):
                        valid_components = []
                        if (
                            mark.cross_bbox is not None
                            and component_validation
                            and component_validation["cross"]["accepted"]
                        ):
                            valid_components.append(
                                mark.model_copy(
                                    update={
                                        "mark_type": "cross",
                                        "bbox": mark.cross_bbox,
                                        "cross_bbox": mark.cross_bbox,
                                        "circle_bbox": None,
                                    }
                                )
                            )
                        if (
                            mark.circle_bbox is not None
                            and component_validation
                            and component_validation["circle"]["accepted"]
                        ):
                            circle_id = (
                                mark.mark_id
                                if not valid_components
                                else next_fallback_mark_id
                            )
                            if valid_components:
                                next_fallback_mark_id += 1
                            valid_components.append(
                                mark.model_copy(
                                    update={
                                        "mark_id": circle_id,
                                        "mark_type": "circle",
                                        "bbox": mark.circle_bbox,
                                        "cross_bbox": None,
                                        "circle_bbox": mark.circle_bbox,
                                    }
                                )
                            )
                        if len(valid_components) == 2 and (
                            not components_within_union or not components_pairable
                        ):
                            fallback_marks = valid_components
                            fallback_type = "split_components"
                        elif len(valid_components) == 1:
                            fallback_marks = valid_components
                            fallback_type = valid_components[0].mark_type
                        elif not valid_components and diagnostic["accepted"]:
                            fallback_marks = [
                                mark.model_copy(
                                    update={
                                        "mark_type": "mixed",
                                        "cross_bbox": None,
                                        "circle_bbox": None,
                                    }
                                )
                            ]
                            fallback_type = "mixed"
                if mark.confidence < confidence_threshold:
                    accepted = False
                    reason = "low_confidence"
                elif fallback_marks:
                    accepted = True
                    reason = "accepted_with_component_fallback"
                elif (
                    not component_accepted
                    or components_within_union is False
                    or components_pairable is False
                ):
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
                    "red_pixel_min_ratio": diagnostic["red_pixel_min_ratio"],
                    "accepted": accepted,
                    "reason": reason,
                    "mark_type": mark.mark_type,
                }
                if component_validation is not None:
                    mark_diagnostic["component_validation"] = component_validation
                if components_within_union is not None:
                    mark_diagnostic["components_within_union"] = (
                        components_within_union
                    )
                if component_pair_distance_ratio is not None:
                    mark_diagnostic["component_pair_distance_ratio"] = round(
                        component_pair_distance_ratio,
                        6,
                    )
                    mark_diagnostic["component_pair_max_distance_ratio"] = (
                        component_pair_max_distance_ratio
                    )
                    mark_diagnostic["components_pairable"] = components_pairable
                if fallback_type is not None:
                    mark_diagnostic["fallback_type"] = fallback_type
                diagnostics.append(mark_diagnostic)
                if accepted:
                    valid.extend(fallback_marks or [mark])
                else:
                    rejected.append(mark.mark_id)
    except (Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ErrorMarkImageInvalid("Error mark image is invalid") from exc
    return valid, rejected, diagnostics
