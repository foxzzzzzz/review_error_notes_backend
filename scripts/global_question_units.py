"""Deterministic page-global question units for offline diagnostics."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np


def validate_bbox(value) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must contain four coordinates")
    bbox = [round(float(item), 6) for item in value]
    if not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
        raise ValueError("bbox must be normalized and non-empty")
    return bbox


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    descriptions = config.get("_descriptions")
    runtime_keys = set(config) - {"_descriptions"}
    if not isinstance(descriptions, dict) or set(descriptions) != runtime_keys:
        raise ValueError("config descriptions must match runtime keys")
    if int(config["anchor_candidate_top_k"]) not in (1, 2, 3):
        raise ValueError("anchor_candidate_top_k must be between 1 and 3")
    return config


def normalize_ocr_lines(ocr_lines: list[dict], config: dict) -> list[dict]:
    accepted = []
    for index, line in enumerate(ocr_lines):
        if line.get("bbox") is None:
            continue
        if float(line.get("confidence", 0.0)) < float(config["ocr_min_confidence"]):
            continue
        accepted.append(
            {
                "ocr_line_id": int(line.get("ocr_line_id", index)),
                "bbox": validate_bbox(line["bbox"]),
                "text": str(line.get("text", "")),
                "confidence": round(float(line["confidence"]), 6),
            }
        )
    return sorted(
        accepted,
        key=lambda item: (
            round((item["bbox"][1] + item["bbox"][3]) / 2, 6),
            round((item["bbox"][0] + item["bbox"][2]) / 2, 6),
            item["ocr_line_id"],
        ),
    )


def _cluster_positions(
    positions: np.ndarray, page_extent: int, merge_distance_ratio: float
) -> list[float]:
    if positions.size == 0:
        return []
    max_gap = max(1, round(page_extent * merge_distance_ratio))
    clusters = [[int(positions[0])]]
    for position in positions[1:]:
        value = int(position)
        if value - clusters[-1][-1] <= max_gap:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [round(float(np.median(cluster)) / page_extent, 6) for cluster in clusters]


def detect_layout_evidence(
    image_bgr: np.ndarray,
    config: dict,
    evidence_mode: str = "combined",
) -> dict:
    if evidence_mode not in {"combined", "grid_only", "ocr_only"}:
        raise ValueError("unsupported evidence mode")
    height, width = image_bgr.shape[:2]
    if evidence_mode == "ocr_only":
        return {
            "horizontal_lines": [],
            "vertical_lines": [],
            "components": [],
            "seeds": [],
            "typical_question_width_ratio": round(
                float(config["fallback_question_width_ratio"]), 6
            ),
            "typical_grid_height_ratio": round(
                float(config["fallback_question_height_ratio"]), 6
            ),
        }
    blue, green, red = cv2.split(image_bgr.astype(np.int16))
    red_mask = (
        (red >= int(config["red_min_channel"]))
        & (red - np.maximum(green, blue) >= int(config["red_min_excess"]))
    ).astype(np.uint8) * 255
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, round(width * float(config["horizontal_line_kernel_width_ratio"]))), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(3, round(height * float(config["vertical_line_kernel_height_ratio"])))),
    )
    horizontal = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, vertical_kernel)
    min_length = float(config["line_min_length_ratio"])
    horizontal_positions = np.flatnonzero(
        np.count_nonzero(horizontal, axis=1) >= max(1, round(width * min_length))
    )
    vertical_positions = np.flatnonzero(
        np.count_nonzero(vertical, axis=0) >= max(1, round(height * min_length))
    )
    merge_ratio = float(config["line_merge_distance_ratio"])
    joined = cv2.dilate(
        cv2.bitwise_or(horizontal, vertical),
        np.ones(
            (int(config["layout_dilation_pixels"]),) * 2,
            dtype=np.uint8,
        ),
    )
    contours, _ = cv2.findContours(
        joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    components = []
    for contour in contours:
        left, top, component_width, component_height = cv2.boundingRect(contour)
        width_ratio = component_width / width
        height_ratio = component_height / height
        if (
            float(config["layout_component_min_width_ratio"])
            <= width_ratio
            <= float(config["layout_component_max_width_ratio"])
            and float(config["layout_component_min_height_ratio"])
            <= height_ratio
            <= float(config["layout_component_max_height_ratio"])
        ):
            components.append(
                [
                    left / width,
                    top / height,
                    (left + component_width) / width,
                    (top + component_height) / height,
                ]
            )
    components.sort(key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))
    typical_widths = [
        bbox[2] - bbox[0]
        for bbox in components
        if bbox[2] - bbox[0] < float(config["layout_typical_max_width_ratio"])
    ]
    typical_heights = [
        bbox[3] - bbox[1]
        for bbox in components
        if bbox[3] - bbox[1] < float(config["layout_typical_max_height_ratio"])
    ]
    typical_width = (
        statistics.median(typical_widths)
        if typical_widths
        else float(config["fallback_question_width_ratio"])
    )
    typical_height = (
        statistics.median(typical_heights)
        if typical_heights
        else float(config["fallback_question_height_ratio"])
    )
    seeds = []
    for bbox in components:
        component_width = bbox[2] - bbox[0]
        component_height = bbox[3] - bbox[1]
        column_count = (
            max(1, round(component_width / typical_width))
            if component_width
            > typical_width * float(config["layout_split_width_multiplier"])
            else 1
        )
        row_count = (
            max(1, round(component_height / typical_height))
            if component_height
            > typical_height * float(config["layout_split_height_multiplier"])
            else 1
        )
        for row_index in range(row_count):
            for column_index in range(column_count):
                seeds.append(
                    [
                        bbox[0] + component_width * column_index / column_count,
                        bbox[1] + component_height * row_index / row_count,
                        bbox[0]
                        + component_width * (column_index + 1) / column_count,
                        bbox[1] + component_height * (row_index + 1) / row_count,
                    ]
                )
    return {
        "horizontal_lines": _cluster_positions(horizontal_positions, height, merge_ratio),
        "vertical_lines": _cluster_positions(vertical_positions, width, merge_ratio),
        "components": [[round(value, 6) for value in bbox] for bbox in components],
        "seeds": [[round(value, 6) for value in bbox] for bbox in seeds],
        "typical_question_width_ratio": round(typical_width, 6),
        "typical_grid_height_ratio": round(typical_height, 6),
    }


def _ocr_centers(ocr_lines: list[dict], axis: int) -> list[float]:
    return sorted((line["bbox"][axis] + line["bbox"][axis + 2]) / 2 for line in ocr_lines)


def _fallback_boundaries(ocr_lines: list[dict], axis: int) -> list[float]:
    centers = _ocr_centers(ocr_lines, axis)
    if not centers:
        return [0.0, 1.0]
    if len(centers) == 1:
        return [0.0, 1.0]
    midpoints = [(first + second) / 2 for first, second in zip(centers, centers[1:])]
    return [0.0, *midpoints, 1.0]


def _bands(boundaries: list[float]) -> list[list[float]]:
    unique = sorted({round(max(0.0, min(1.0, value)), 6) for value in boundaries})
    return [[unique[index], unique[index + 1]] for index in range(len(unique) - 1)
            if unique[index + 1] - unique[index] > 1e-6]


def build_page_bands(
    image_shape: tuple[int, ...],
    ocr_lines: list[dict],
    layout_evidence: dict,
    config: dict,
) -> dict:
    del image_shape, config
    horizontal = layout_evidence["horizontal_lines"]
    vertical = layout_evidence["vertical_lines"]
    row_boundaries = horizontal if len(horizontal) >= 2 else _fallback_boundaries(ocr_lines, 1)
    column_boundaries = vertical if len(vertical) >= 2 else _fallback_boundaries(ocr_lines, 0)
    return {
        "row_bands": _bands(row_boundaries),
        "column_bands": _bands(column_boundaries),
        "grid_missing": len(horizontal) < 2 or len(vertical) < 2,
    }


def _contains_center(bbox: list[float], line_bbox: list[float]) -> bool:
    center_x = (line_bbox[0] + line_bbox[2]) / 2
    center_y = (line_bbox[1] + line_bbox[3]) / 2
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def _context_bbox(bbox: list[float], config: dict) -> list[float]:
    padding_x = float(config["unit_context_padding_x_ratio"])
    padding_y = float(config["unit_context_padding_y_ratio"])
    return [
        round(max(0.0, bbox[0] - padding_x), 6),
        round(max(0.0, bbox[1] - padding_y), 6),
        round(min(1.0, bbox[2] + padding_x), 6),
        round(min(1.0, bbox[3] + padding_y), 6),
    ]


def materialize_units(bands: dict, ocr_lines: list[dict], config: dict) -> list[dict]:
    units = []
    has_ocr = bool(ocr_lines)
    for row_index, row in enumerate(bands["row_bands"], start=1):
        column_number = 0
        for column in bands["column_bands"]:
            bbox = [column[0], row[0], column[1], row[1]]
            memberships = [
                line["ocr_line_id"] for line in ocr_lines
                if _contains_center(bbox, line["bbox"])
            ]
            if has_ocr and not memberships:
                continue
            column_number += 1
            risks = []
            if not has_ocr:
                risks.append("ocr_missing")
            if bands["grid_missing"]:
                risks.append("grid_missing")
            units.append(
                {
                    "question_unit_id": f"U-S01-R{row_index:02d}-C{column_number:02d}",
                    "unit_bbox": [round(value, 6) for value in bbox],
                    "context_bbox": _context_bbox(bbox, config),
                    "ocr_line_ids": sorted(memberships),
                    "layout_evidence": "grid" if not bands["grid_missing"] else "ocr",
                    "risk_flags": risks,
                }
            )
    return units


def _question_bbox_for_seed(
    seed_bbox: list[float], typical_width: float, typical_height: float, config: dict
) -> list[float]:
    center_x = (seed_bbox[0] + seed_bbox[2]) / 2
    center_y = (seed_bbox[1] + seed_bbox[3]) / 2
    width = max(
        typical_width,
        min(
            seed_bbox[2] - seed_bbox[0],
            typical_width * float(config["question_width_cap_multiplier"]),
        ),
    )
    half_width = width * float(config["question_horizontal_half_width_multiplier"])
    half_height = typical_height * float(config["question_height_multiplier"]) / 2
    return [
        round(max(0.0, center_x - half_width), 6),
        round(max(0.0, center_y - half_height), 6),
        round(min(1.0, center_x + half_width), 6),
        round(min(1.0, center_y + half_height), 6),
    ]


def _materialize_component_units(
    evidence: dict, ocr_lines: list[dict], config: dict
) -> list[dict]:
    typical_width = float(evidence["typical_question_width_ratio"])
    typical_height = float(evidence["typical_grid_height_ratio"])
    seed_items = []
    for seed in evidence["seeds"]:
        seed_bbox = validate_bbox(seed)
        center_y = (seed_bbox[1] + seed_bbox[3]) / 2
        center_x = (seed_bbox[0] + seed_bbox[2]) / 2
        seed_items.append((center_y, center_x, seed_bbox))
    seed_items.sort(key=lambda item: (item[0], item[1], item[2]))
    rows = []
    row_tolerance = max(typical_height * 0.75, 1e-6)
    for item in seed_items:
        if not rows or abs(item[0] - statistics.median(value[0] for value in rows[-1])) > row_tolerance:
            rows.append([item])
        else:
            rows[-1].append(item)
    units = []
    for row_index, row in enumerate(rows, start=1):
        for column_index, (_, _, seed_bbox) in enumerate(
            sorted(row, key=lambda item: (item[1], item[0], item[2])), start=1
        ):
            unit_bbox = _question_bbox_for_seed(
                seed_bbox, typical_width, typical_height, config
            )
            memberships = sorted(
                line["ocr_line_id"]
                for line in ocr_lines
                if _contains_center(unit_bbox, line["bbox"])
            )
            risks = [] if ocr_lines else ["ocr_missing"]
            if not evidence["components"]:
                risks.append("grid_missing")
            units.append(
                {
                    "question_unit_id": f"U-S01-R{row_index:02d}-C{column_index:02d}",
                    "unit_bbox": unit_bbox,
                    "context_bbox": _context_bbox(unit_bbox, config),
                    "ocr_line_ids": memberships,
                    "layout_evidence": "grid_component",
                    "risk_flags": risks,
                }
            )
    return units


def build_global_question_units(
    image_bgr: np.ndarray,
    ocr_lines: list[dict],
    config: dict,
    evidence_mode: str = "combined",
) -> dict:
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("image_bgr must be a color image")
    normalized = normalize_ocr_lines(ocr_lines, config)
    evidence = detect_layout_evidence(image_bgr, config, evidence_mode=evidence_mode)
    seeds = list(evidence["seeds"])
    if evidence_mode != "grid_only":
        for line in normalized:
            bbox = line["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if (
                float(config["ocr_line_min_width_ratio"])
                <= width
                <= float(config["ocr_line_max_width_ratio"])
                and float(config["ocr_line_min_height_ratio"])
                <= height
                <= float(config["ocr_line_max_height_ratio"])
            ):
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                coverage_ratio = float(config["ocr_seed_layout_coverage_ratio"])
                covered_by_layout = any(
                    _contains_point(seed, center_x, center_y)
                    and seed[2] - seed[0] >= width * coverage_ratio
                    and seed[3] - seed[1] >= height * coverage_ratio
                    for seed in evidence["seeds"]
                )
                if not covered_by_layout:
                    seeds.append(bbox)
    evidence["seeds"] = sorted(
        {tuple(round(value, 6) for value in seed) for seed in seeds},
        key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]),
    )
    evidence["seeds"] = [list(seed) for seed in evidence["seeds"]]
    bands = build_page_bands(image_bgr.shape, normalized, evidence, config)
    if evidence["seeds"]:
        units = _materialize_component_units(evidence, normalized, config)
    else:
        units = materialize_units(bands, normalized, config)
    return {
        "units": units,
        "row_bands": bands["row_bands"],
        "column_bands": bands["column_bands"],
        "layout_evidence": evidence,
    }


def _contains_point(bbox: list[float], x: float, y: float, margin: float = 0.0) -> bool:
    return (
        bbox[0] - margin <= x <= bbox[2] + margin
        and bbox[1] - margin <= y <= bbox[3] + margin
    )


def rank_units_for_anchor(units: list[dict], anchor: dict, config: dict) -> list[dict]:
    anchor_bbox = validate_bbox(anchor["bbox"])
    anchor_x = (anchor_bbox[0] + anchor_bbox[2]) / 2
    anchor_y = (anchor_bbox[1] + anchor_bbox[3]) / 2
    margin = float(config["anchor_context_margin_ratio"])
    max_distance = float(config["anchor_max_center_distance_ratio"])
    ranked = []
    for unit in units:
        unit_bbox = validate_bbox(unit["unit_bbox"])
        context_bbox = validate_bbox(unit["context_bbox"])
        unit_x = (unit_bbox[0] + unit_bbox[2]) / 2
        unit_y = (unit_bbox[1] + unit_bbox[3]) / 2
        center_distance = math.hypot(unit_x - anchor_x, unit_y - anchor_y)
        anchor_in_unit = _contains_point(unit_bbox, anchor_x, anchor_y)
        anchor_in_context = _contains_point(context_bbox, anchor_x, anchor_y, margin)
        if not anchor_in_context and center_distance > max_distance:
            continue
        distance_score = max(0.0, 1.0 - center_distance / max(max_distance, 1e-9))
        score = (
            (2.0 if anchor_in_unit else 0.0)
            + (1.0 if anchor_in_context else 0.0)
            + distance_score
        )
        ranked.append(
            {
                "question_unit_id": str(unit["question_unit_id"]),
                "score": round(score, 6),
                "anchor_in_unit": anchor_in_unit,
                "anchor_in_context": anchor_in_context,
                "center_distance": round(center_distance, 6),
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["question_unit_id"]))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def map_anchors_to_units(
    units: list[dict], anchors: list[dict], config: dict
) -> dict:
    anchor_candidates = {}
    unassigned = []
    top_k = int(config["anchor_candidate_top_k"])
    for anchor in sorted(anchors, key=lambda item: int(item["cross_id"])):
        cross_id = int(anchor["cross_id"])
        ranked = rank_units_for_anchor(units, anchor, config)[:top_k]
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        anchor_candidates[str(cross_id)] = ranked
        if not ranked:
            unassigned.append(
                {"cross_id": cross_id, "reason": "no_unit_within_distance"}
            )
    return {
        "anchor_candidates": anchor_candidates,
        "unassigned_anchors": unassigned,
    }


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _match_metrics(unit_bbox: list[float], truth_bbox: list[float]) -> tuple[float, float]:
    intersection = _intersection_area(unit_bbox, truth_bbox)
    union = _bbox_area(unit_bbox) + _bbox_area(truth_bbox) - intersection
    truth_area = _bbox_area(truth_bbox)
    return (
        intersection / union if union > 0 else 0.0,
        intersection / truth_area if truth_area > 0 else 0.0,
    )


def audit_fixed_units(
    selected_units: list[dict], truth_regions: list[dict], config: dict
) -> dict:
    min_iou = float(config["truth_match_min_iou"])
    min_coverage = float(config["truth_match_min_coverage"])
    intrusion_coverage = float(config["sibling_intrusion_min_coverage"])
    truth = [
        {
            "truth_id": str(item["truth_id"]),
            "bbox": validate_bbox(item["source_bbox_normalized"]),
        }
        for item in truth_regions
    ]
    matches_by_truth = {item["truth_id"]: [] for item in truth}
    atomic_matches_by_truth = {item["truth_id"]: [] for item in truth}
    matches_by_unit = {}
    false_units = []
    sibling_intrusions = []
    non_atomic_units = []
    for unit in selected_units:
        unit_id = str(unit["question_unit_id"])
        unit_bbox = validate_bbox(unit["unit_bbox"])
        matched = []
        overlaps = []
        for item in truth:
            iou, coverage = _match_metrics(unit_bbox, item["bbox"])
            if iou >= min_iou or coverage >= min_coverage:
                matched.append(item["truth_id"])
                matches_by_truth[item["truth_id"]].append(unit_id)
            if coverage >= intrusion_coverage:
                overlaps.append(item["truth_id"])
        matches_by_unit[unit_id] = matched
        if not matched:
            false_units.append(unit_id)
        elif len(matched) == 1 and len(overlaps) == 1:
            atomic_matches_by_truth[matched[0]].append(unit_id)
        else:
            non_atomic_units.append(unit_id)
        if matched and any(truth_id not in matched for truth_id in overlaps):
            sibling_intrusions.append(unit_id)
    matched_truth_ids = sorted(
        truth_id for truth_id, unit_ids in matches_by_truth.items() if unit_ids
    )
    missed_truth_ids = sorted(set(matches_by_truth) - set(matched_truth_ids))
    duplicate_truth_ids = sorted(
        truth_id for truth_id, unit_ids in matches_by_truth.items() if len(unit_ids) > 1
    )
    atomic_matched_truth_ids = sorted(
        truth_id for truth_id, unit_ids in atomic_matches_by_truth.items() if unit_ids
    )
    atomic_missed_truth_ids = sorted(
        set(atomic_matches_by_truth) - set(atomic_matched_truth_ids)
    )
    return {
        "matched_truth_ids": matched_truth_ids,
        "missed_truth_ids": missed_truth_ids,
        "truth_recall": round(
            len(matched_truth_ids) / len(truth) if truth else 1.0, 6
        ),
        "false_unit_ids": sorted(false_units),
        "duplicate_truth_ids": duplicate_truth_ids,
        "sibling_intrusion_unit_ids": sorted(set(sibling_intrusions)),
        "atomic_matched_truth_ids": atomic_matched_truth_ids,
        "atomic_missed_truth_ids": atomic_missed_truth_ids,
        "atomic_truth_recall": round(
            len(atomic_matched_truth_ids) / len(truth) if truth else 1.0, 6
        ),
        "non_atomic_unit_ids": sorted(non_atomic_units),
        "unit_truth_matches": matches_by_unit,
    }


def compare_unit_candidates_to_truth(
    units: list[dict],
    anchor_mapping: dict,
    truth_regions: list[dict],
    config: dict,
) -> dict:
    selected_ids = {
        candidate["question_unit_id"]
        for candidates in anchor_mapping["anchor_candidates"].values()
        for candidate in candidates
    }
    selected = [
        unit for unit in units if unit["question_unit_id"] in selected_ids
    ]
    return audit_fixed_units(selected, truth_regions, config)
