"""Diagnose deterministic OCR/layout candidates with one numbered LLM selection per page.

The experiment replays existing CV + LLM1 anchors. The model may select only a
candidate ID or reject an anchor; it never generates question geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict, Field


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.local_ocr_verification import RapidOCRVerifier
from app.services.vision_recognition import (
    MiniMaxVisionClient,
    prepare_image_data_url,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name(
    "numbered_question_selector_config.json"
)


NUMBERED_SELECTION_PROMPT = """你是小学作业错题候选选择器。当前图片是一张编号局部拼图；每个Q编号块中，绿色框是本地OCR/版面算法生成的确定性题目候选，蓝色框是需要核验的红叉锚点。输入JSON列出每个cross_id只允许选择的candidate_id。

目标：判断每个红叉是否为老师表示学生作答错误的真实红叉，并从允许列表中选择与它对应的最小完整独立作答单元。

要求：
1. 每个cross_id必须且只能返回一次，不得新增、遗漏、合并或重排。
2. decision=selected时，selected_candidate_id必须来自该cross_id的allowed_candidate_ids；不得选择其他锚点的候选。
3. decision=none表示图片中的蓝色锚点不是真实判错红叉，或附近没有任何候选与其对应；此时selected_candidate_id必须为null。
4. decision=uncertain只用于确实无法区分两个候选的情况；此时selected_candidate_id必须为null，不得猜测。
5. 绿色框才是最终会保存的题目范围；绿色框外的宽上下文只用于观察。不得因为上下文中包含另一道题而选择错误候选。
6. 红叉是主证据；附近红圈可作为辅助证据，用于判断红叉对应哪道题，但红圈、批注或印刷红色方格不能单独创建错题。
7. 不识别题目文字，不返回题目内容，不得返回或生成任何 bbox、坐标或新候选。
8. 只返回严格JSON，不要解释或Markdown。

返回格式：{"selections":[{"cross_id":0,"decision":"selected","selected_candidate_id":"Q0","confidence":0.95},{"cross_id":1,"decision":"none","selected_candidate_id":null,"confidence":0.9}]}。

输入候选：__CANDIDATES__
"""


class NumberedSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cross_id: int
    decision: Literal["selected", "none", "uncertain"]
    selected_candidate_id: str | None = None
    confidence: float = Field(ge=0, le=1)


class NumberedSelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    selections: list[NumberedSelection]


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _validate_bbox(value) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox must contain four values")
    bbox = [float(item) for item in value]
    if not all(0 <= item <= 1 for item in bbox):
        raise ValueError("bbox values must be normalized")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("bbox must have positive area")
    return bbox


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _bbox_iou(first: list[float], second: list[float]) -> float:
    intersection = _intersection_area(first, second)
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union else 0.0


def bbox_contains(container: list[float], candidate: list[float]) -> bool:
    return (
        container[0] <= candidate[0]
        and container[1] <= candidate[1]
        and container[2] >= candidate[2]
        and container[3] >= candidate[3]
    )


def _bbox_union(first: list[float], second: list[float]) -> list[float]:
    return [
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    ]


def _point_bbox_distance(bbox: list[float], x: float, y: float) -> float:
    return math.hypot(
        max(bbox[0] - x, 0.0, x - bbox[2]),
        max(bbox[1] - y, 0.0, y - bbox[3]),
    )


def _contains_point(bbox: list[float], x: float, y: float, margin=0.0) -> bool:
    return (
        bbox[0] - margin <= x <= bbox[2] + margin
        and bbox[1] - margin <= y <= bbox[3] + margin
    )


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    descriptions = config.get("_descriptions")
    if not isinstance(descriptions, dict):
        raise ValueError("config requires _descriptions")
    runtime_keys = set(config) - {"_descriptions"}
    if runtime_keys != set(descriptions):
        raise ValueError("every runtime parameter must have a description")
    if int(config["candidate_top_k_per_anchor"]) < 1:
        raise ValueError("candidate_top_k_per_anchor must be positive")
    half_width_multipliers = config["question_horizontal_half_width_multipliers"]
    if (
        not isinstance(half_width_multipliers, list)
        or len(half_width_multipliers) != 2
        or any(float(value) <= 0 for value in half_width_multipliers)
    ):
        raise ValueError(
            "question_horizontal_half_width_multipliers requires two positive values"
        )
    for name in (
        "candidate_dedup_iou_threshold",
        "truth_anchor_margin_ratio",
        "truth_match_min_iou",
        "truth_match_min_coverage",
        "sibling_intrusion_min_coverage",
    ):
        if not 0 <= float(config[name]) <= 1:
            raise ValueError(f"{name} must be between zero and one")
    return config


def _layout_seeds(pixels: np.ndarray, config: dict) -> tuple[list[dict], float, float]:
    height, width = pixels.shape[:2]
    red = pixels[:, :, 0].astype(np.int16)
    green = pixels[:, :, 1].astype(np.int16)
    blue = pixels[:, :, 2].astype(np.int16)
    mask = (
        (red >= int(config["layout_red_min_channel"]))
        & (red - np.maximum(green, blue) >= int(config["layout_red_min_excess"]))
    ).astype(np.uint8) * 255
    horizontal_width = max(
        3, round(width * float(config["layout_horizontal_kernel_width_ratio"]))
    )
    vertical_height = max(
        3, round(height * float(config["layout_vertical_kernel_height_ratio"]))
    )
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_width, 1)),
    )
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_height)),
    )
    dilation = int(config["layout_dilation_pixels"])
    joined = cv2.dilate(
        cv2.bitwise_or(horizontal, vertical),
        np.ones((dilation, dilation), np.uint8),
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
    for bbox in sorted(components, key=lambda item: (item[1], item[0])):
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
        for row in range(row_count):
            for column in range(column_count):
                seeds.append(
                    {
                        "source": "layout",
                        "bbox": [
                            bbox[0] + component_width * column / column_count,
                            bbox[1] + component_height * row / row_count,
                            bbox[0]
                            + component_width * (column + 1) / column_count,
                            bbox[1] + component_height * (row + 1) / row_count,
                        ],
                    }
                )
    return seeds, typical_width, typical_height


def _question_bbox_for_seed(
    seed_bbox: list[float],
    typical_width: float,
    typical_height: float,
    half_width_multiplier: float,
    config: dict,
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
    half_width = width * half_width_multiplier
    half_height = (
        typical_height * float(config["question_height_multiplier"]) / 2
    )
    return [
        round(max(0.0, center_x - half_width), 6),
        round(max(0.0, center_y - half_height), 6),
        round(min(1.0, center_x + half_width), 6),
        round(min(1.0, center_y + half_height), 6),
    ]


def _context_bbox(
    question_bbox: list[float], anchor_bbox: list[float], config: dict
) -> list[float]:
    union = _bbox_union(question_bbox, anchor_bbox)
    return [
        round(max(0.0, union[0] - float(config["context_padding_x_ratio"])), 6),
        round(max(0.0, union[1] - float(config["context_padding_y_ratio"])), 6),
        round(min(1.0, union[2] + float(config["context_padding_x_ratio"])), 6),
        round(min(1.0, union[3] + float(config["context_padding_y_ratio"])), 6),
    ]


def build_numbered_candidates(
    *, image_path: Path, anchors: list[dict], ocr_lines: list[dict], config: dict
) -> dict:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    pixels = np.asarray(image)
    seeds, typical_width, typical_height = _layout_seeds(pixels, config)
    for line in ocr_lines:
        bbox = _validate_bbox(line.get("bbox"))
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if (
            float(line.get("confidence", 0))
            >= float(config["ocr_line_confidence_threshold"])
            and float(config["ocr_line_min_width_ratio"])
            <= width
            <= float(config["ocr_line_max_width_ratio"])
            and float(config["ocr_line_min_height_ratio"])
            <= height
            <= float(config["ocr_line_max_height_ratio"])
        ):
            seeds.append({"source": "ocr", "bbox": bbox})
    if not seeds:
        seeds = [
            {"source": "anchor_fallback", "bbox": _validate_bbox(anchor["bbox"])}
            for anchor in anchors
        ]
    half_width_multipliers = [
        float(value)
        for value in config["question_horizontal_half_width_multipliers"]
    ]
    if len(half_width_multipliers) != 2:
        raise ValueError("question_horizontal_half_width_multipliers requires two values")
    boundary_variants = ("compact", "standard")
    ranked_seeds = []
    for seed in seeds:
        for variant_rank, (variant, multiplier) in enumerate(
            zip(boundary_variants, half_width_multipliers)
        ):
            ranked_seeds.append(
                {
                    "source": seed["source"],
                    "bbox": seed["bbox"],
                    "boundary_variant": variant,
                    "variant_rank": variant_rank,
                    "question_bbox": _question_bbox_for_seed(
                        seed["bbox"],
                        typical_width,
                        typical_height,
                        multiplier,
                        config,
                    ),
                }
            )

    candidates = []
    anchor_candidates = {}
    top_k = int(config["candidate_top_k_per_anchor"])
    dedup_threshold = float(config["candidate_dedup_iou_threshold"])
    for anchor in sorted(anchors, key=lambda item: int(item["cross_id"])):
        cross_id = int(anchor["cross_id"])
        anchor_bbox = _validate_bbox(anchor["bbox"])
        center_x = (anchor_bbox[0] + anchor_bbox[2]) / 2
        center_y = (anchor_bbox[1] + anchor_bbox[3]) / 2
        ranked = sorted(
            ranked_seeds,
            key=lambda seed: (
                _point_bbox_distance(seed["bbox"], center_x, center_y),
                0 if seed["source"] == "layout" else 1,
                seed["variant_rank"],
                _bbox_area(seed["bbox"]),
                seed["bbox"],
            ),
        )[:top_k]
        assigned_ids = []
        for seed in ranked:
            question_bbox = seed["question_bbox"]
            existing = next(
                (
                    candidate
                    for candidate in candidates
                    if _bbox_iou(candidate["question_bbox"], question_bbox)
                    >= dedup_threshold
                ),
                None,
            )
            if existing is None:
                existing = {
                    "candidate_id": f"Q{len(candidates)}",
                    "seed_source": seed["source"],
                    "boundary_variant": seed["boundary_variant"],
                    "question_bbox": question_bbox,
                    "context_bbox": _context_bbox(
                        question_bbox, anchor_bbox, config
                    ),
                    "cross_ids": [],
                }
                candidates.append(existing)
            else:
                existing["context_bbox"] = _context_bbox(
                    existing["context_bbox"], anchor_bbox, config
                )
            if cross_id not in existing["cross_ids"]:
                existing["cross_ids"].append(cross_id)
            if existing["candidate_id"] not in assigned_ids:
                assigned_ids.append(existing["candidate_id"])
        anchor_candidates[cross_id] = assigned_ids
    return {
        "candidates": candidates,
        "anchor_candidates": anchor_candidates,
        "seed_count": len(seeds),
        "typical_question_width_ratio": round(typical_width, 6),
        "typical_grid_height_ratio": round(typical_height, 6),
    }


def audit_selections(selections: list[dict], allowed: dict[int, list[str]]) -> dict:
    known_candidates = {
        candidate_id for candidate_ids in allowed.values() for candidate_id in candidate_ids
    }
    accepted = []
    violations = []
    counts = {}
    for selection in selections:
        cross_id = int(selection["cross_id"])
        counts[cross_id] = counts.get(cross_id, 0) + 1
        candidate_id = selection.get("selected_candidate_id")
        decision = selection.get("decision")
        reason = None
        if cross_id not in allowed:
            reason = "unknown_anchor"
        elif candidate_id is not None and candidate_id not in known_candidates:
            reason = "unknown_candidate_id"
        elif decision == "selected" and candidate_id not in allowed[cross_id]:
            reason = "candidate_not_allowed_for_anchor"
        elif decision != "selected" and candidate_id is not None:
            reason = "non_selected_decision_has_candidate"
        elif decision == "selected" and candidate_id is None:
            reason = "selected_decision_missing_candidate"
        if reason:
            violations.append({"cross_id": cross_id, "reason": reason})
        else:
            accepted.append(
                {
                    "cross_id": cross_id,
                    "decision": decision,
                    "selected_candidate_id": candidate_id,
                    "confidence": float(selection["confidence"]),
                }
            )
    for cross_id in allowed:
        if counts.get(cross_id, 0) == 0:
            violations.append({"cross_id": cross_id, "reason": "missing_anchor"})
        elif counts[cross_id] > 1:
            violations.append({"cross_id": cross_id, "reason": "duplicate_anchor"})
    return {
        "valid": not violations,
        "accepted": accepted,
        "violations": violations,
    }


def build_selected_events(accepted: list[dict], candidates: list[dict]) -> list[dict]:
    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    grouped = {}
    for selection in accepted:
        if selection["decision"] != "selected":
            continue
        grouped.setdefault(selection["selected_candidate_id"], []).append(selection)
    events = []
    for candidate_id, selections in grouped.items():
        candidate = candidate_by_id[candidate_id]
        events.append(
            {
                "event_id": len(events),
                "candidate_id": candidate_id,
                "cross_ids": sorted(item["cross_id"] for item in selections),
                "question_bbox": list(candidate["question_bbox"]),
                "confidence": max(item["confidence"] for item in selections),
            }
        )
    return events


def build_candidate_oracle_events(candidates: list[dict]) -> list[dict]:
    return [
        {
            "event_id": index,
            "candidate_id": candidate["candidate_id"],
            "cross_ids": sorted(candidate["cross_ids"]),
            "question_bbox": list(candidate["question_bbox"]),
            "confidence": 1.0,
        }
        for index, candidate in enumerate(candidates)
    ]


def compare_selected_events_to_truth(
    *, anchors: list[dict], events: list[dict], truth_regions: list[dict], config: dict
) -> dict:
    anchor_by_id = {int(anchor["cross_id"]): anchor for anchor in anchors}
    margin = float(config["truth_anchor_margin_ratio"])
    min_iou = float(config["truth_match_min_iou"])
    min_coverage = float(config["truth_match_min_coverage"])
    intrusion_threshold = float(config["sibling_intrusion_min_coverage"])
    matched_truth_ids = []
    truth_event_ids = {}
    for truth in truth_regions:
        truth_bbox = _validate_bbox(truth["source_bbox_normalized"])
        matched_events = []
        for event in events:
            event_bbox = _validate_bbox(event["question_bbox"])
            overlap = _intersection_area(event_bbox, truth_bbox)
            coverage = overlap / _bbox_area(truth_bbox)
            has_truth_anchor = any(
                cross_id in anchor_by_id
                and _contains_point(
                    truth_bbox,
                    (anchor_by_id[cross_id]["bbox"][0]
                     + anchor_by_id[cross_id]["bbox"][2]) / 2,
                    (anchor_by_id[cross_id]["bbox"][1]
                     + anchor_by_id[cross_id]["bbox"][3]) / 2,
                    margin,
                )
                for cross_id in event["cross_ids"]
            )
            if has_truth_anchor and coverage >= min_coverage and _bbox_iou(
                event_bbox, truth_bbox
            ) >= min_iou:
                matched_events.append(event["event_id"])
        if matched_events:
            matched_truth_ids.append(truth["truth_id"])
        truth_event_ids[truth["truth_id"]] = matched_events
    matched_event_ids = {
        event_id for event_ids in truth_event_ids.values() for event_id in event_ids
    }
    intrusion_event_ids = []
    for event in events:
        overlapping_truth_count = sum(
            _intersection_area(event["question_bbox"], truth["source_bbox_normalized"])
            / _bbox_area(truth["source_bbox_normalized"])
            >= intrusion_threshold
            for truth in truth_regions
        )
        if overlapping_truth_count > 1:
            intrusion_event_ids.append(event["event_id"])
    truth_count = len(truth_regions)
    return {
        "truth_count": truth_count,
        "matched_truth_ids": matched_truth_ids,
        "matched_truth_count": len(matched_truth_ids),
        "truth_recall": round(len(matched_truth_ids) / truth_count, 6)
        if truth_count
        else None,
        "missed_truth_ids": [
            truth["truth_id"]
            for truth in truth_regions
            if truth["truth_id"] not in matched_truth_ids
        ],
        "false_event_ids": [
            event["event_id"]
            for event in events
            if event["event_id"] not in matched_event_ids
        ],
        "duplicate_truth_ids": [
            truth_id for truth_id, event_ids in truth_event_ids.items() if len(event_ids) > 1
        ],
        "sibling_intrusion_event_ids": intrusion_event_ids,
    }


def write_candidate_montage(
    *, image_path: Path, output_path: Path, anchors: list[dict], candidates: list[dict], config: dict
) -> None:
    with Image.open(image_path) as source:
        page = ImageOps.exif_transpose(source).convert("RGB")
    tile_width = int(config["montage_tile_width"])
    tile_height = int(config["montage_tile_height"])
    label_height = int(config["montage_label_height"])
    columns = int(config["montage_columns"])
    rows = max(1, math.ceil(len(candidates) / columns))
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    anchor_by_id = {int(anchor["cross_id"]): anchor for anchor in anchors}
    font = ImageFont.load_default()
    for index, candidate in enumerate(candidates):
        context = candidate["context_bbox"]
        pixel_context = (
            round(context[0] * page.width),
            round(context[1] * page.height),
            round(context[2] * page.width),
            round(context[3] * page.height),
        )
        crop = page.crop(pixel_context)
        draw = ImageDraw.Draw(crop)

        def local_pixels(bbox):
            context_width = max(context[2] - context[0], 1e-9)
            context_height = max(context[3] - context[1], 1e-9)
            return (
                round((bbox[0] - context[0]) / context_width * crop.width),
                round((bbox[1] - context[1]) / context_height * crop.height),
                round((bbox[2] - context[0]) / context_width * crop.width),
                round((bbox[3] - context[1]) / context_height * crop.height),
            )

        draw.rectangle(local_pixels(candidate["question_bbox"]), outline="lime", width=6)
        for cross_id in candidate["cross_ids"]:
            draw.rectangle(
                local_pixels(anchor_by_id[cross_id]["bbox"]), outline="blue", width=5
            )
        available_height = tile_height - label_height
        crop.thumbnail((tile_width, available_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(
            crop,
            ((tile_width - crop.width) // 2, label_height + (available_height - crop.height) // 2),
        )
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.rectangle((0, 0, tile_width - 1, tile_height - 1), outline="black")
        tile_draw.text(
            (8, 8),
            f"{candidate['candidate_id']} / C{','.join(map(str, candidate['cross_ids']))}",
            fill="black",
            font=font,
        )
        montage.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    montage.save(output_path, quality=int(config["montage_jpeg_quality"]))


def _resolve_anchor_path(root: Path, label: str) -> Path:
    candidates = (
        root / label / "cross-anchor-experiment" / "confirmed-crosses.json",
        root / "pages" / label / "cross-anchor-experiment" / "confirmed-crosses.json",
    )
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one confirmed-crosses replay for {label}")
    return matches[0]


def _parse_images(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"image must use label=path: {value}")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not label or not path.is_file():
            raise ValueError(f"invalid image: {value}")
        parsed.append((label, path))
    return parsed


def _load_truth(path: Path, labels: list[str]) -> dict[str, list[dict]]:
    pages = json.loads(path.read_text(encoding="utf-8")).get("pages")
    if not isinstance(pages, dict):
        raise ValueError("truth JSON must contain pages")
    result = {}
    for label in labels:
        regions = pages.get(label, {}).get("regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"truth page has no regions: {label}")
        result[label] = regions
    return result


def _ocr_verifier() -> RapidOCRVerifier:
    return RapidOCRVerifier(
        enabled=settings.LOCAL_OCR_ENABLED,
        library_version=settings.LOCAL_OCR_VERSION,
        engine_name=settings.LOCAL_OCR_ENGINE,
        model_version=settings.LOCAL_OCR_MODEL_VERSION,
        model_type=settings.LOCAL_OCR_MODEL_TYPE,
        model_path=settings.LOCAL_OCR_MODEL_PATH,
        max_pixels=settings.QUESTION_IMAGE_MAX_PIXELS,
        line_confidence_threshold=settings.LOCAL_OCR_LINE_CONFIDENCE_THRESHOLD,
        min_effective_characters=settings.LOCAL_OCR_MIN_EFFECTIVE_CHARACTERS,
        support_similarity_threshold=settings.LOCAL_OCR_SUPPORT_SIMILARITY_THRESHOLD,
        contradiction_similarity_threshold=settings.LOCAL_OCR_CONTRADICTION_SIMILARITY_THRESHOLD,
    )


def _write_report(path: Path, summaries: list[dict]) -> None:
    lines = [
        "# 编号候选选择型LLM2诊断",
        "",
        "> LLM只能选择本地确定性窄框编号或返回none/uncertain，不生成坐标。",
        "",
        "| 图片 | 真值 | 锚点 | 编号候选 | 候选上限命中 | 候选上限召回 | 已选锚点 | none | uncertain | 去重事件 | LLM选择命中 | LLM选择召回 | 误报事件 | 兄弟题侵入 | 语义异常 | OCR耗时(ms) | 版面耗时(ms) | LLM耗时(ms) | 总耗时(ms) | LLM请求 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {label} | {truth_count} | {anchor_count} | {candidate_count} | "
            "{candidate_oracle_matched_truth_count} | {candidate_oracle_truth_recall} | "
            "{selected_anchor_count} | {none_count} | {uncertain_count} | "
            "{event_count} | {matched_truth_count} | {truth_recall} | "
            "{false_event_count} | {sibling_intrusion_event_count} | "
            "{semantic_violation_count} | {ocr_ms} | {layout_ms} | {llm_ms} | "
            "{total_ms} | {llm_request_count} |".format(**item)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_page(
    *, label: str, image_path: Path, anchor_path: Path, truth_regions: list[dict],
    config: dict, output_root: Path, subject: str, offline_only: bool
) -> dict:
    started = time.perf_counter()
    page_dir = output_root / label
    page_dir.mkdir(parents=True, exist_ok=False)
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    for anchor in anchors:
        _validate_bbox(anchor.get("bbox"))
    ocr_started = time.perf_counter()
    page_ocr = _ocr_verifier().recognize_page(
        str(image_path), int(config["ocr_full_page_max_edge"])
    )
    ocr_ms = (time.perf_counter() - ocr_started) * 1000
    if page_ocr.status != "available":
        raise RuntimeError(f"page OCR unavailable: {page_ocr.error_code or page_ocr.status}")
    ocr_lines = [line.model_dump(mode="json") for line in page_ocr.lines]
    layout_started = time.perf_counter()
    proposal = build_numbered_candidates(
        image_path=image_path, anchors=anchors, ocr_lines=ocr_lines, config=config
    )
    layout_ms = (time.perf_counter() - layout_started) * 1000
    _write_json(page_dir / "numbered-candidates.json", proposal)
    montage_path = page_dir / "numbered-candidate-montage.jpg"
    write_candidate_montage(
        image_path=image_path,
        output_path=montage_path,
        anchors=anchors,
        candidates=proposal["candidates"],
        config=config,
    )
    allowed = proposal["anchor_candidates"]
    llm_ms = 0.0
    llm_request_count = 0
    selections = []
    llm_error = None
    if not offline_only and anchors:
        payload = [
            {
                "cross_id": cross_id,
                "allowed_candidate_ids": candidate_ids,
                "subject_hint": subject,
            }
            for cross_id, candidate_ids in allowed.items()
        ]
        prompt = NUMBERED_SELECTION_PROMPT.replace(
            "__CANDIDATES__", json.dumps(payload, ensure_ascii=False, indent=2)
        )
        events = []
        client = MiniMaxVisionClient.from_settings()
        client.diagnostic_event_sink = events.append
        diagnostic = {
            "operation": "numbered_question_candidate_selection",
            "anchor_count": len(anchors),
            "candidate_count": len(proposal["candidates"]),
        }
        llm_started = time.perf_counter()
        llm_request_count = 1
        try:
            result = client._request(
                {
                    "prompt": prompt,
                    "image_url": prepare_image_data_url(
                        str(montage_path), client.max_edge, client.jpeg_quality, diagnostic
                    ),
                },
                NumberedSelectionResult,
                diagnostic,
            )
            selections = [item.model_dump(mode="json") for item in result.selections]
        except Exception as exc:
            llm_error = {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "message": str(exc),
                "diagnostic": getattr(exc, "diagnostic", None),
            }
        llm_ms = (time.perf_counter() - llm_started) * 1000
        _write_json(page_dir / "llm-events.json", events)
        _write_json(page_dir / "llm-error.json", llm_error)
    _write_json(page_dir / "llm-selections.json", selections)
    audit = audit_selections(selections, allowed) if not offline_only else {
        "valid": True, "accepted": [], "violations": []
    }
    events = build_selected_events(audit["accepted"], proposal["candidates"])
    candidate_oracle = compare_selected_events_to_truth(
        anchors=anchors,
        events=build_candidate_oracle_events(proposal["candidates"]),
        truth_regions=truth_regions,
        config=config,
    )
    comparison = compare_selected_events_to_truth(
        anchors=anchors, events=events, truth_regions=truth_regions, config=config
    )
    _write_json(page_dir / "selection-audit.json", audit)
    _write_json(page_dir / "selected-events.json", events)
    _write_json(page_dir / "candidate-oracle-comparison.json", candidate_oracle)
    _write_json(page_dir / "oracle-comparison.json", comparison)
    selected = [item for item in audit["accepted"] if item["decision"] == "selected"]
    summary = {
        "label": label,
        "status": "offline_only" if offline_only else ("llm_error" if llm_error else "completed"),
        "anchor_count": len(anchors),
        "candidate_count": len(proposal["candidates"]),
        "selected_anchor_count": len(selected),
        "none_count": sum(item["decision"] == "none" for item in audit["accepted"]),
        "uncertain_count": sum(item["decision"] == "uncertain" for item in audit["accepted"]),
        "event_count": len(events),
        "candidate_oracle_matched_truth_count": candidate_oracle["matched_truth_count"],
        "candidate_oracle_truth_recall": candidate_oracle["truth_recall"],
        "candidate_oracle_missed_truth_ids": candidate_oracle["missed_truth_ids"],
        "false_event_count": len(comparison["false_event_ids"]),
        "sibling_intrusion_event_count": len(comparison["sibling_intrusion_event_ids"]),
        "semantic_violation_count": len(audit["violations"]),
        "ocr_ms": round(ocr_ms, 2),
        "layout_ms": round(layout_ms, 2),
        "llm_ms": round(llm_ms, 2),
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
        "llm_request_count": llm_request_count,
        **comparison,
    }
    _write_json(page_dir / "summary.json", summary)
    return summary


def main(arguments=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Images in label=/absolute/path form")
    parser.add_argument("--anchors-root", required=True, type=Path)
    parser.add_argument("--truth-regions", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--subject", default="chinese")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(arguments)
    images = _parse_images(args.images)
    labels = [label for label, _ in images]
    if len(set(labels)) != len(labels):
        parser.error("image labels must be unique")
    anchor_root = args.anchors_root.expanduser().resolve()
    if not anchor_root.is_dir():
        parser.error("--anchors-root must be an existing directory")
    config = load_config(args.config.expanduser().resolve())
    truth = _load_truth(args.truth_regions.expanduser().resolve(), labels)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for label, image_path in images:
        try:
            summaries.append(
                run_page(
                    label=label,
                    image_path=image_path,
                    anchor_path=_resolve_anchor_path(anchor_root, label),
                    truth_regions=truth[label],
                    config=config,
                    output_root=output,
                    subject=args.subject,
                    offline_only=args.offline_only,
                )
            )
        except Exception as exc:
            error_dir = output / label
            error_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                error_dir / "page-error.json",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            summaries.append(
                {
                    "label": label, "status": "page_error", "truth_count": len(truth[label]),
                    "anchor_count": 0, "candidate_count": 0, "selected_anchor_count": 0,
                    "none_count": 0, "uncertain_count": 0, "event_count": 0,
                    "candidate_oracle_matched_truth_count": 0,
                    "candidate_oracle_truth_recall": 0.0,
                    "matched_truth_count": 0, "truth_recall": 0.0,
                    "false_event_count": 0, "sibling_intrusion_event_count": 0,
                    "semantic_violation_count": 0, "ocr_ms": 0.0, "layout_ms": 0.0,
                    "llm_ms": 0.0, "total_ms": 0.0, "llm_request_count": 0,
                }
            )
    aggregate = {
        "experiment": "numbered_question_candidate_selection",
        "offline_only": args.offline_only,
        "labels": labels,
        "llm_request_count": sum(item["llm_request_count"] for item in summaries),
        "truth_count": sum(item["truth_count"] for item in summaries),
        "matched_truth_count": sum(item["matched_truth_count"] for item in summaries),
        "page_summaries": summaries,
    }
    aggregate["truth_recall"] = round(
        aggregate["matched_truth_count"] / aggregate["truth_count"], 6
    )
    _write_json(output / "summary.json", aggregate)
    _write_report(output / "comparison-report.md", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
