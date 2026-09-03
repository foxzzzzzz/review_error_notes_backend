"""Diagnose deterministic question boundaries and conservative anchor filtering.

The experiment replays existing CV + LLM1 anchors. Local rules select question
geometry; one optional LLM request per page only classifies anchor shapes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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


ANCHOR_VERIFICATION_PROMPT = """你是小学作业批改红叉核验器。图片是红色标记局部放大拼图，每块顶部标有C编号，蓝框只是本地算法给出的待核验区域，不代表其中一定有红叉。

你的唯一任务是逐个判断蓝框内的主要红色形状，不判断题目内容，不选择题目区域。

判定标准：
1. real_cross：必须能直接看到两条方向相反、彼此相交的红色斜笔画，整体构成清晰X。只有这种情况才能选择real_cross。
2. not_cross：红圈或椭圆、页码圆圈、单条斜线、下划线、老师批注、文字笔画、印刷红色方格，或者蓝框内没有清晰X。
3. uncertain：图片确实模糊、笔画被遮挡或只能看到红叉的一部分，无法可靠区分real_cross与not_cross。不得为了完成任务而猜测。
4. 蓝框是候选提示而不是答案；页面中可以大多数都不是红叉，也可以全部都是红叉。必须独立检查每个C编号，禁止默认全部确认。
5. 红圈只能帮助理解上下文，红圈本身不是红叉。红圈与单条斜线相邻也不能算X。
6. 每个输入cross_id必须且只能返回一次，不得新增、遗漏、合并或重排。
7. visual_evidence必须选择实际观察到的形状：two_intersecting_red_diagonal_strokes、circle_or_oval、single_or_nonintersecting_stroke、printed_grid_or_text、insufficient_detail、other_red_mark。
8. 不返回题目内容、候选编号、bbox或坐标。只返回严格JSON，不要解释或Markdown。

返回格式：{"verifications":[{"cross_id":0,"decision":"real_cross","visual_evidence":"two_intersecting_red_diagonal_strokes","confidence":0.95},{"cross_id":1,"decision":"not_cross","visual_evidence":"circle_or_oval","confidence":0.9}]}。

输入锚点：__ANCHORS__
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


class AnchorVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cross_id: int
    decision: Literal["real_cross", "not_cross", "uncertain"]
    visual_evidence: Literal[
        "two_intersecting_red_diagonal_strokes",
        "circle_or_oval",
        "single_or_nonintersecting_stroke",
        "printed_grid_or_text",
        "insufficient_detail",
        "other_red_mark",
    ]
    confidence: float = Field(ge=0, le=1)


class AnchorVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    verifications: list[AnchorVerification]


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
        "atomic_duplicate_iou_threshold",
        "atomic_duplicate_ocr_jaccard_threshold",
        "atomic_same_row_center_y_ratio",
        "atomic_same_column_center_x_ratio",
        "atomic_horizontal_min_center_separation_ratio",
        "atomic_horizontal_partition_overlap_ratio",
        "atomic_vertical_min_center_separation_ratio",
        "atomic_ocr_row_merge_gap_height_multiplier",
    ):
        if not 0 <= float(config[name]) <= 1:
            raise ValueError(f"{name} must be between zero and one")
    for name in (
        "local_recheck_center_window_ratio",
        "local_recheck_min_center_red_ratio",
        "local_recheck_bottom_page_number_min_y_ratio",
        "local_recheck_page_number_max_distance_ratio",
        "anchor_montage_crop_padding_ratio",
    ):
        if not 0 <= float(config[name]) <= 1:
            raise ValueError(f"{name} must be between zero and one")
    if float(config["atomic_vertical_clip_min_width_multiplier"]) <= 0:
        raise ValueError(
            "atomic_vertical_clip_min_width_multiplier must be positive"
        )
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


def _numeric_ocr_near_anchor(
    anchor_bbox: list[float], ocr_lines: list[dict], max_distance: float
) -> str | None:
    anchor_x = (anchor_bbox[0] + anchor_bbox[2]) / 2
    anchor_y = (anchor_bbox[1] + anchor_bbox[3]) / 2
    for line in ocr_lines:
        text = re.sub(r"\s+", "", str(line.get("text", "")))
        if not re.fullmatch(r"\d{1,3}", text):
            continue
        bbox = _validate_bbox(line.get("bbox"))
        line_x = (bbox[0] + bbox[2]) / 2
        line_y = (bbox[1] + bbox[3]) / 2
        if math.hypot(line_x - anchor_x, line_y - anchor_y) <= max_distance:
            return text
    return None


def assess_local_anchor_geometry(
    *, image_path: Path, anchors: list[dict], ocr_lines: list[dict], config: dict
) -> list[dict]:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    pixels = np.asarray(image, dtype=np.int16)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    red_mask = (red >= int(config["local_recheck_red_min_channel"])) & (
        red - np.maximum(green, blue) >= int(config["local_recheck_red_min_excess"])
    )
    center_ratio = float(config["local_recheck_center_window_ratio"])
    min_center_red = float(config["local_recheck_min_center_red_ratio"])
    bottom_y = float(config["local_recheck_bottom_page_number_min_y_ratio"])
    page_number_distance = float(
        config["local_recheck_page_number_max_distance_ratio"]
    )
    assessments = []
    for anchor in sorted(anchors, key=lambda item: int(item["cross_id"])):
        bbox = _validate_bbox(anchor["bbox"])
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        half_width = max((bbox[2] - bbox[0]) * center_ratio, 1 / image.width)
        half_height = max((bbox[3] - bbox[1]) * center_ratio, 1 / image.height)
        left = max(0, round((center_x - half_width) * image.width))
        top = max(0, round((center_y - half_height) * image.height))
        right = min(image.width, round((center_x + half_width) * image.width) + 1)
        bottom = min(image.height, round((center_y + half_height) * image.height) + 1)
        center_red_ratio = float(red_mask[top:bottom, left:right].mean())
        page_number_text = None
        if center_y >= bottom_y:
            page_number_text = _numeric_ocr_near_anchor(
                bbox, ocr_lines, page_number_distance
            )
        if page_number_text is not None:
            decision = "reject"
            reason = "bottom_page_number"
        elif center_red_ratio < min_center_red:
            decision = "reject"
            reason = "insufficient_center_red"
        else:
            decision = "keep"
            reason = "center_red_supported"
        assessments.append(
            {
                "cross_id": int(anchor["cross_id"]),
                "decision": decision,
                "reason": reason,
                "center_red_ratio": round(center_red_ratio, 6),
                "page_number_text": page_number_text,
            }
        )
    return assessments


def audit_anchor_verifications(
    verifications: list[dict], anchor_ids: list[int]
) -> dict:
    expected = set(anchor_ids)
    counts = {cross_id: 0 for cross_id in anchor_ids}
    accepted = []
    violations = []
    for verification in verifications:
        cross_id = int(verification["cross_id"])
        if cross_id not in expected:
            violations.append({"cross_id": cross_id, "reason": "unknown_anchor"})
            continue
        counts[cross_id] += 1
        if (
            verification["decision"] == "real_cross"
            and verification["visual_evidence"]
            != "two_intersecting_red_diagonal_strokes"
        ):
            violations.append(
                {"cross_id": cross_id, "reason": "real_cross_without_cross_evidence"}
            )
            continue
        accepted.append(verification)
    for cross_id in anchor_ids:
        if counts[cross_id] == 0:
            violations.append({"cross_id": cross_id, "reason": "missing_anchor"})
        elif counts[cross_id] > 1:
            violations.append({"cross_id": cross_id, "reason": "duplicate_anchor"})
    return {"valid": not violations, "accepted": accepted, "violations": violations}


def consensus_anchor_filter(
    *, anchor_ids: list[int], local_assessments: list[dict],
    llm_verifications: list[dict]
) -> dict:
    local_by_id = {int(item["cross_id"]): item for item in local_assessments}
    llm_by_id = {int(item["cross_id"]): item for item in llm_verifications}
    rejected = [
        cross_id
        for cross_id in anchor_ids
        if local_by_id.get(cross_id, {}).get("decision") == "reject"
        and llm_by_id.get(cross_id, {}).get("decision") == "not_cross"
    ]
    return {
        "kept_cross_ids": [cross_id for cross_id in anchor_ids if cross_id not in rejected],
        "rejected_cross_ids": rejected,
    }


def build_deterministic_events(
    *, anchors: list[dict], candidates: list[dict], allowed: dict[int, list[str]],
    kept_cross_ids: list[int]
) -> list[dict]:
    anchor_by_id = {int(anchor["cross_id"]): anchor for anchor in anchors}
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    grouped: dict[str, list[int]] = {}
    for cross_id in kept_cross_ids:
        candidate_ids = allowed[cross_id]
        candidate_id = next(
            (
                item
                for item in candidate_ids
                if candidate_by_id[item]["boundary_variant"] == "standard"
            ),
            candidate_ids[0],
        )
        grouped.setdefault(candidate_id, []).append(cross_id)
    events = []
    for candidate_id, cross_ids in grouped.items():
        candidate = candidate_by_id[candidate_id]
        events.append(
            {
                "event_id": len(events),
                "candidate_id": candidate_id,
                "cross_ids": sorted(cross_ids),
                "question_bbox": list(candidate["question_bbox"]),
                "confidence": max(
                    float(anchor_by_id[cross_id].get("confidence", 0))
                    for cross_id in cross_ids
                ),
            }
        )
    return events


def _ocr_line_membership(
    bbox: list[float], ocr_lines: list[dict]
) -> set[int]:
    members = set()
    for index, line in enumerate(ocr_lines):
        line_bbox = _validate_bbox(line["bbox"])
        center_x = (line_bbox[0] + line_bbox[2]) / 2
        center_y = (line_bbox[1] + line_bbox[3]) / 2
        if _contains_point(bbox, center_x, center_y):
            members.add(index)
    return members


def _set_jaccard(first: set[int], second: set[int]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def build_atomic_question_events(
    *, events: list[dict], ocr_lines: list[dict], config: dict,
    anchors: list[dict] | None = None
) -> tuple[list[dict], dict]:
    duplicate_iou = float(config["atomic_duplicate_iou_threshold"])
    duplicate_ocr_jaccard = float(
        config["atomic_duplicate_ocr_jaccard_threshold"]
    )
    merged = []
    duplicate_groups = []
    for event in events:
        event_bbox = _validate_bbox(event["question_bbox"])
        event_ocr = _ocr_line_membership(event_bbox, ocr_lines)
        duplicate = next(
            (
                item
                for item in merged
                if _bbox_iou(item["question_bbox"], event_bbox) >= duplicate_iou
                and _set_jaccard(item["ocr_line_ids"], event_ocr)
                >= duplicate_ocr_jaccard
            ),
            None,
        )
        if duplicate is None:
            merged.append(
                {
                    "event_id": len(merged),
                    "candidate_id": event["candidate_id"],
                    "cross_ids": sorted(event["cross_ids"]),
                    "question_bbox": event_bbox,
                    "confidence": float(event["confidence"]),
                    "source_event_ids": [int(event["event_id"])],
                    "ocr_line_ids": event_ocr,
                }
            )
            continue
        duplicate["cross_ids"] = sorted(
            set(duplicate["cross_ids"]) | set(event["cross_ids"])
        )
        duplicate["confidence"] = max(
            duplicate["confidence"], float(event["confidence"])
        )
        duplicate["source_event_ids"].append(int(event["event_id"]))
        duplicate["ocr_line_ids"] |= event_ocr

    duplicate_groups = [
        item["source_event_ids"]
        for item in merged
        if len(item["source_event_ids"]) > 1
    ]
    anchor_by_id = {
        int(anchor["cross_id"]): anchor for anchor in (anchors or [])
    }
    geometry = []
    for item in merged:
        bbox = list(item["question_bbox"])
        geometry.append(
            {
                "event": item,
                "bbox": bbox,
                "center_x": (bbox[0] + bbox[2]) / 2,
                "center_y": (bbox[1] + bbox[3]) / 2,
                "width": bbox[2] - bbox[0],
                "height": bbox[3] - bbox[1],
            }
        )
    median_width = statistics.median(
        item["width"] for item in geometry
    ) if geometry else 0.0
    row_tolerance = float(config["atomic_same_row_center_y_ratio"])
    column_tolerance = float(config["atomic_same_column_center_x_ratio"])
    horizontal_separation = float(
        config["atomic_horizontal_min_center_separation_ratio"]
    )
    horizontal_overlap = float(
        config["atomic_horizontal_partition_overlap_ratio"]
    )
    vertical_separation = float(
        config["atomic_vertical_min_center_separation_ratio"]
    )
    wide_multiplier = float(
        config["atomic_vertical_clip_min_width_multiplier"]
    )
    partitioned = set()
    for current in geometry:
        event = current["event"]
        bbox = event["question_bbox"]
        left_centers = []
        right_centers = []
        for other in geometry:
            if other is current:
                continue
            min_width = min(current["width"], other["width"])
            min_height = min(current["height"], other["height"])
            if (
                abs(other["center_y"] - current["center_y"])
                <= row_tolerance * min_height
                and abs(other["center_x"] - current["center_x"])
                >= horizontal_separation * min_width
            ):
                if other["center_x"] < current["center_x"]:
                    left_centers.append(other["center_x"])
                else:
                    right_centers.append(other["center_x"])
        original = list(bbox)
        if left_centers:
            boundary = (current["center_x"] + max(left_centers)) / 2
            bbox[0] = max(
                bbox[0], boundary - current["width"] * horizontal_overlap
            )
        if right_centers:
            boundary = (current["center_x"] + min(right_centers)) / 2
            bbox[2] = min(
                bbox[2], boundary + current["width"] * horizontal_overlap
            )
        if median_width and current["width"] > median_width * wide_multiplier:
            above = []
            for other in geometry:
                if other is current or other["center_y"] >= current["center_y"]:
                    continue
                min_width = min(current["width"], other["width"])
                min_height = min(current["height"], other["height"])
                if (
                    current["center_y"] - other["center_y"]
                    >= vertical_separation * min_height
                    and abs(other["center_x"] - current["center_x"])
                    <= column_tolerance * min_width
                    and other["bbox"][3] > current["bbox"][1]
                ):
                    above.append(other)
            if above:
                nearest = max(above, key=lambda item: item["center_y"])
                bbox[1] = max(bbox[1], nearest["bbox"][3])
        for cross_id in event["cross_ids"]:
            anchor = anchor_by_id.get(cross_id)
            if anchor is None:
                continue
            anchor_bbox = _validate_bbox(anchor["bbox"])
            anchor_x = (anchor_bbox[0] + anchor_bbox[2]) / 2
            anchor_y = (anchor_bbox[1] + anchor_bbox[3]) / 2
            if not _contains_point(original, anchor_x, anchor_y):
                continue
            bbox[0] = min(bbox[0], anchor_x)
            bbox[1] = min(bbox[1], anchor_y)
            bbox[2] = max(bbox[2], anchor_x)
            bbox[3] = max(bbox[3], anchor_y)
        if bbox != original:
            partitioned.add(int(event["event_id"]))

    refined = []
    for item in merged:
        refined.append(
            {
                "event_id": len(refined),
                "candidate_id": item["candidate_id"],
                "cross_ids": item["cross_ids"],
                "question_bbox": [round(value, 6) for value in item["question_bbox"]],
                "confidence": item["confidence"],
            }
        )
    event_ocr_row_audits = []
    multi_ocr_row_event_ids = []
    anchor_outside_event_ids = []
    median_ocr_height = (
        statistics.median(
            _validate_bbox(line["bbox"])[3] - _validate_bbox(line["bbox"])[1]
            for line in ocr_lines
        )
        if ocr_lines
        else 0.0
    )
    final_row_gap = median_ocr_height * float(
        config["atomic_ocr_row_merge_gap_height_multiplier"]
    )
    for event in refined:
        bbox = event["question_bbox"]
        line_centers = []
        for line in ocr_lines:
            line_bbox = _validate_bbox(line["bbox"])
            center_x = (line_bbox[0] + line_bbox[2]) / 2
            center_y = (line_bbox[1] + line_bbox[3]) / 2
            if _contains_point(bbox, center_x, center_y):
                line_centers.append(center_y)
        line_centers.sort()
        row_group_count = 0
        previous_center = None
        for center_y in line_centers:
            if previous_center is None or center_y - previous_center > final_row_gap:
                row_group_count += 1
            previous_center = center_y
        multi_row_risk = row_group_count > 2
        event_ocr_row_audits.append(
            {
                "event_id": int(event["event_id"]),
                "ocr_line_count": len(line_centers),
                "ocr_row_group_count": row_group_count,
                "multi_row_risk": multi_row_risk,
            }
        )
        if multi_row_risk:
            multi_ocr_row_event_ids.append(int(event["event_id"]))
        if any(
            cross_id in anchor_by_id
            and not _contains_point(
                bbox,
                (
                    _validate_bbox(anchor_by_id[cross_id]["bbox"])[0]
                    + _validate_bbox(anchor_by_id[cross_id]["bbox"])[2]
                )
                / 2,
                (
                    _validate_bbox(anchor_by_id[cross_id]["bbox"])[1]
                    + _validate_bbox(anchor_by_id[cross_id]["bbox"])[3]
                )
                / 2,
            )
            for cross_id in event["cross_ids"]
        ):
            anchor_outside_event_ids.append(int(event["event_id"]))
    return refined, {
        "ocr_duplicate_groups": duplicate_groups,
        "partitioned_event_ids": sorted(partitioned),
        "event_ocr_row_audits": event_ocr_row_audits,
        "multi_ocr_row_event_ids": multi_ocr_row_event_ids,
        "anchor_outside_event_ids": anchor_outside_event_ids,
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


def write_anchor_verification_montage(
    *, image_path: Path, output_path: Path, anchors: list[dict], config: dict
) -> None:
    with Image.open(image_path) as source:
        page = ImageOps.exif_transpose(source).convert("RGB")
    tile_width = int(config["anchor_montage_tile_width"])
    tile_height = int(config["anchor_montage_tile_height"])
    label_height = int(config["anchor_montage_label_height"])
    columns = int(config["anchor_montage_columns"])
    padding = float(config["anchor_montage_crop_padding_ratio"])
    rows = max(1, math.ceil(len(anchors) / columns))
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    font = ImageFont.load_default()
    for index, anchor in enumerate(sorted(anchors, key=lambda item: int(item["cross_id"]))):
        bbox = _validate_bbox(anchor["bbox"])
        context = [
            max(0.0, bbox[0] - padding),
            max(0.0, bbox[1] - padding),
            min(1.0, bbox[2] + padding),
            min(1.0, bbox[3] + padding),
        ]
        pixel_context = (
            round(context[0] * page.width),
            round(context[1] * page.height),
            round(context[2] * page.width),
            round(context[3] * page.height),
        )
        crop = page.crop(pixel_context)
        context_width = max(context[2] - context[0], 1e-9)
        context_height = max(context[3] - context[1], 1e-9)
        local_bbox = (
            round((bbox[0] - context[0]) / context_width * crop.width),
            round((bbox[1] - context[1]) / context_height * crop.height),
            round((bbox[2] - context[0]) / context_width * crop.width),
            round((bbox[3] - context[1]) / context_height * crop.height),
        )
        ImageDraw.Draw(crop).rectangle(local_bbox, outline="blue", width=5)
        available_height = tile_height - label_height
        crop.thumbnail((tile_width, available_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(
            crop,
            ((tile_width - crop.width) // 2, label_height + (available_height - crop.height) // 2),
        )
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.rectangle((0, 0, tile_width - 1, tile_height - 1), outline="black")
        tile_draw.text((8, 8), f"C{int(anchor['cross_id'])}", fill="black", font=font)
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
        "# 本地确定性边界与红叉保守复检诊断",
        "",
        "> 题目边界由本地规则确定；MiniMax每页只核验一次红叉真假，不选择候选、不生成坐标。",
        "",
        "| 图片 | 真值 | 锚点 | 候选上限召回 | 本地拒绝锚点 | LLM真叉 | LLM拒绝 | LLM不确定 | 全保留召回 | 全保留误报 | 本地召回 | 本地误报 | 原子题召回 | 原子题误报 | 原子题事件 | OCR重复合并 | 多OCR行风险 | 锚点在题框外 | 兄弟题侵入 | LLM单独召回 | LLM单独误报 | 一致拒绝召回 | 一致拒绝误报 | 语义异常 | OCR耗时(ms) | 版面耗时(ms) | 本地复检(ms) | 原子化(ms) | LLM耗时(ms) | 总耗时(ms) | LLM请求 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {label} | {truth_count} | {anchor_count} | {candidate_oracle_truth_recall} | "
            "{local_rejected_anchor_count} | {llm_real_cross_count} | "
            "{llm_not_cross_count} | {llm_uncertain_count} | "
            "{all_truth_recall} | {all_false_event_count} | "
            "{local_truth_recall} | {local_false_event_count} | "
            "{atomic_truth_recall} | {atomic_false_event_count} | "
            "{atomic_event_count} | {atomic_ocr_duplicate_group_count} | "
            "{atomic_multi_ocr_row_event_count} | "
            "{atomic_anchor_outside_event_count} | "
            "{atomic_sibling_intrusion_event_count} | "
            "{llm_truth_recall} | {llm_false_event_count} | "
            "{consensus_truth_recall} | {consensus_false_event_count} | "
            "{semantic_violation_count} | {ocr_ms} | {layout_ms} | "
            "{local_recheck_ms} | {atomic_ms} | {llm_ms} | {total_ms} | "
            "{llm_request_count} |".format(**item)
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
    anchor_ids = [int(anchor["cross_id"]) for anchor in anchors]
    local_started = time.perf_counter()
    local_assessments = assess_local_anchor_geometry(
        image_path=image_path,
        anchors=anchors,
        ocr_lines=ocr_lines,
        config=config,
    )
    local_recheck_ms = (time.perf_counter() - local_started) * 1000
    _write_json(page_dir / "local-anchor-assessments.json", local_assessments)
    anchor_montage_path = page_dir / "anchor-verification-montage.jpg"
    write_anchor_verification_montage(
        image_path=image_path,
        output_path=anchor_montage_path,
        anchors=anchors,
        config=config,
    )
    llm_ms = 0.0
    llm_request_count = 0
    verifications = []
    llm_error = None
    if not offline_only and anchors:
        payload = [
            {"cross_id": cross_id, "subject_hint": subject}
            for cross_id in anchor_ids
        ]
        prompt = ANCHOR_VERIFICATION_PROMPT.replace(
            "__ANCHORS__", json.dumps(payload, ensure_ascii=False, indent=2)
        )
        llm_events = []
        client = MiniMaxVisionClient.from_settings()
        client.diagnostic_event_sink = llm_events.append
        diagnostic = {
            "operation": "anchor_shape_verification",
            "anchor_count": len(anchors),
        }
        llm_started = time.perf_counter()
        llm_request_count = 1
        try:
            result = client._request(
                {
                    "prompt": prompt,
                    "image_url": prepare_image_data_url(
                        str(anchor_montage_path),
                        client.max_edge,
                        client.jpeg_quality,
                        diagnostic,
                    ),
                },
                AnchorVerificationResult,
                diagnostic,
            )
            verifications = [
                item.model_dump(mode="json") for item in result.verifications
            ]
        except Exception as exc:
            llm_error = {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "message": str(exc),
                "diagnostic": getattr(exc, "diagnostic", None),
            }
        llm_ms = (time.perf_counter() - llm_started) * 1000
        _write_json(page_dir / "llm-events.json", llm_events)
        _write_json(page_dir / "llm-error.json", llm_error)
    _write_json(page_dir / "llm-anchor-verifications.json", verifications)
    audit = (
        audit_anchor_verifications(verifications, anchor_ids)
        if not offline_only
        else {"valid": True, "accepted": [], "violations": []}
    )
    _write_json(page_dir / "llm-anchor-audit.json", audit)
    local_kept_ids = [
        item["cross_id"]
        for item in local_assessments
        if item["decision"] == "keep"
    ]
    llm_by_id = {int(item["cross_id"]): item for item in audit["accepted"]}
    llm_kept_ids = [
        cross_id
        for cross_id in anchor_ids
        if llm_by_id.get(cross_id, {}).get("decision") != "not_cross"
    ]
    consensus = consensus_anchor_filter(
        anchor_ids=anchor_ids,
        local_assessments=local_assessments,
        llm_verifications=audit["accepted"],
    )
    all_events = build_deterministic_events(
        anchors=anchors,
        candidates=proposal["candidates"],
        allowed=allowed,
        kept_cross_ids=anchor_ids,
    )
    local_events = build_deterministic_events(
        anchors=anchors,
        candidates=proposal["candidates"],
        allowed=allowed,
        kept_cross_ids=local_kept_ids,
    )
    llm_events = build_deterministic_events(
        anchors=anchors,
        candidates=proposal["candidates"],
        allowed=allowed,
        kept_cross_ids=llm_kept_ids,
    )
    consensus_events = build_deterministic_events(
        anchors=anchors,
        candidates=proposal["candidates"],
        allowed=allowed,
        kept_cross_ids=consensus["kept_cross_ids"],
    )
    atomic_started = time.perf_counter()
    atomic_events, atomic_audit = build_atomic_question_events(
        events=local_events,
        ocr_lines=ocr_lines,
        config=config,
        anchors=anchors,
    )
    atomic_ms = (time.perf_counter() - atomic_started) * 1000
    candidate_oracle = compare_selected_events_to_truth(
        anchors=anchors,
        events=build_candidate_oracle_events(proposal["candidates"]),
        truth_regions=truth_regions,
        config=config,
    )
    comparisons = {
        name: compare_selected_events_to_truth(
            anchors=anchors,
            events=events,
            truth_regions=truth_regions,
            config=config,
        )
        for name, events in (
            ("all", all_events),
            ("local", local_events),
            ("atomic", atomic_events),
            ("llm", llm_events),
            ("consensus", consensus_events),
        )
    }
    _write_json(page_dir / "all-anchor-events.json", all_events)
    _write_json(page_dir / "local-filtered-events.json", local_events)
    _write_json(page_dir / "local-atomic-events.json", atomic_events)
    _write_json(page_dir / "local-atomic-audit.json", atomic_audit)
    _write_json(page_dir / "llm-filtered-events.json", llm_events)
    _write_json(page_dir / "consensus-filtered-events.json", consensus_events)
    _write_json(page_dir / "anchor-filter-consensus.json", consensus)
    _write_json(page_dir / "candidate-oracle-comparison.json", candidate_oracle)
    for name, comparison in comparisons.items():
        _write_json(page_dir / f"{name}-comparison.json", comparison)
    summary = {
        "label": label,
        "status": "offline_only" if offline_only else ("llm_error" if llm_error else "completed"),
        "anchor_count": len(anchors),
        "candidate_count": len(proposal["candidates"]),
        "candidate_oracle_matched_truth_count": candidate_oracle["matched_truth_count"],
        "candidate_oracle_truth_recall": candidate_oracle["truth_recall"],
        "candidate_oracle_missed_truth_ids": candidate_oracle["missed_truth_ids"],
        "local_rejected_anchor_count": len(anchor_ids) - len(local_kept_ids),
        "llm_real_cross_count": sum(
            item["decision"] == "real_cross" for item in audit["accepted"]
        ),
        "llm_not_cross_count": sum(
            item["decision"] == "not_cross" for item in audit["accepted"]
        ),
        "llm_uncertain_count": sum(
            item["decision"] == "uncertain" for item in audit["accepted"]
        ),
        "semantic_violation_count": len(audit["violations"]),
        "ocr_ms": round(ocr_ms, 2),
        "layout_ms": round(layout_ms, 2),
        "local_recheck_ms": round(local_recheck_ms, 2),
        "atomic_ms": round(atomic_ms, 2),
        "llm_ms": round(llm_ms, 2),
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
        "llm_request_count": llm_request_count,
        "truth_count": len(truth_regions),
        "atomic_ocr_duplicate_group_count": len(
            atomic_audit["ocr_duplicate_groups"]
        ),
        "atomic_multi_ocr_row_event_count": len(
            atomic_audit["multi_ocr_row_event_ids"]
        ),
        "atomic_anchor_outside_event_count": len(
            atomic_audit["anchor_outside_event_ids"]
        ),
        "atomic_sibling_intrusion_event_count": len(
            comparisons["atomic"]["sibling_intrusion_event_ids"]
        ),
    }
    for name, comparison in comparisons.items():
        summary[f"{name}_matched_truth_count"] = comparison["matched_truth_count"]
        summary[f"{name}_truth_recall"] = comparison["truth_recall"]
        summary[f"{name}_missed_truth_ids"] = comparison["missed_truth_ids"]
        summary[f"{name}_false_event_count"] = len(comparison["false_event_ids"])
        summary[f"{name}_event_count"] = len(
            {
                "all": all_events,
                "local": local_events,
                "atomic": atomic_events,
                "llm": llm_events,
                "consensus": consensus_events,
            }[name]
        )
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
                    "anchor_count": 0, "candidate_count": 0,
                    "candidate_oracle_matched_truth_count": 0,
                    "candidate_oracle_truth_recall": 0.0,
                    "local_rejected_anchor_count": 0, "llm_real_cross_count": 0,
                    "llm_not_cross_count": 0, "llm_uncertain_count": 0,
                    "all_truth_recall": 0.0, "all_false_event_count": 0,
                    "local_truth_recall": 0.0, "local_false_event_count": 0,
                    "atomic_truth_recall": 0.0, "atomic_false_event_count": 0,
                    "atomic_event_count": 0,
                    "atomic_ocr_duplicate_group_count": 0,
                    "atomic_multi_ocr_row_event_count": 0,
                    "atomic_anchor_outside_event_count": 0,
                    "atomic_sibling_intrusion_event_count": 0,
                    "llm_truth_recall": 0.0, "llm_false_event_count": 0,
                    "consensus_truth_recall": 0.0,
                    "consensus_false_event_count": 0,
                    "semantic_violation_count": 0, "ocr_ms": 0.0,
                    "layout_ms": 0.0, "local_recheck_ms": 0.0,
                    "atomic_ms": 0.0, "llm_ms": 0.0, "total_ms": 0.0,
                    "llm_request_count": 0,
                    "consensus_matched_truth_count": 0,
                    "atomic_matched_truth_count": 0,
                }
            )
    aggregate = {
        "experiment": "deterministic_boundary_local_atomic_filter",
        "offline_only": args.offline_only,
        "labels": labels,
        "llm_request_count": sum(item["llm_request_count"] for item in summaries),
        "truth_count": sum(item["truth_count"] for item in summaries),
        "matched_truth_count": sum(
            item["consensus_matched_truth_count"] for item in summaries
        ),
        "atomic_matched_truth_count": sum(
            item["atomic_matched_truth_count"] for item in summaries
        ),
        "page_summaries": summaries,
    }
    aggregate["truth_recall"] = round(
        aggregate["matched_truth_count"] / aggregate["truth_count"], 6
    )
    aggregate["atomic_truth_recall"] = round(
        aggregate["atomic_matched_truth_count"] / aggregate["truth_count"], 6
    )
    _write_json(output / "summary.json", aggregate)
    _write_report(output / "comparison-report.md", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
