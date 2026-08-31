"""Capture CV and three-stage vision evidence without persisting questions.

The output contains worksheet images and model responses. Store it only in a
restricted diagnostic directory and delete it after analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageOps
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.error_mark_validation import (
    group_red_evidence_regions,
    scan_red_mark_regions,
)
from app.services.vision_recognition import (
    ErrorMark,
    MarkDetectionResult,
    MiniMaxVisionClient,
    prepare_image_data_url,
    recognize_marked_three_stage,
    validate_normalized_bbox,
)


STABLE_EVENT_CONSOLIDATION_PROMPT = """你是小学作业红色批改事件归并器。图片是在原始整页作业上叠加了 P0、P1... 编号框的红色几何标记候选；输入 JSON 同时给出每个 primitive 的类型和整页坐标。

目标：把属于同一次判错的邻近红圈和红叉归并为一个稳定事件，确保一个真实圈叉组合只产生一个 event_id。

要求：
1. 每个 primitive_id 必须且只能出现一次：要么属于某个 event 的 primitive_ids，要么放入 unassigned_primitive_ids。
2. 同一作答单元附近、几何距离相近的 circle 与 cross 应归并为 cross_circle；不得因框大小不同或局部重叠而重复建立事件。
3. 一个 cross_circle 必须且只能包含一个 circle primitive 和一个 cross primitive。任何 event 都不得包含两个 circle 或两个 cross；相邻但属于不同作答单元的标记必须拆成不同 event。
4. 单独 circle、cross、deletion、underline、annotation 或 mixed event 只能包含一个 primitive；无法确定归属的 primitive 放入 unassigned_primitive_ids，不得猜测。
5. 教师文字批注、评语或提示如果不直接标记某个具体作答单元，必须放入 unassigned_primitive_ids，不得建立错误事件。
6. 空间明显分离、跨行或跨作答单元的 circle 与 cross 禁止配对；无法确认同属一个作答单元时保持单标记或放入 unassigned_primitive_ids。
7. bbox、cross_bbox、circle_bbox 均使用原始整页归一化坐标，并覆盖图片中可见的完整几何形状，不得只复制输入中的红线碎片框。无对应图形时必须返回 null，不得返回 [0,0,0,0]。
8. event_type 只能使用 cross_circle、circle、cross、deletion、underline、annotation、mixed；不得使用 cross_only 或 circle_only。
9. event_id 从 0 开始连续且唯一。只返回严格 JSON，不要解释或 Markdown。

返回格式：{"events":[{"event_id":0,"primitive_ids":[0,1],"event_type":"cross_circle","bbox":[0.1,0.2,0.4,0.5],"cross_bbox":[0.3,0.2,0.4,0.35],"circle_bbox":[0.1,0.25,0.35,0.5],"confidence":0.95}],"unassigned_primitive_ids":[]}。

输入 primitives：__PRIMITIVES__
"""


INDEPENDENT_COMPLETE_MARK_PROMPT = """你是小学作业红色批改几何标记检测器。请只根据当前原始整页图片，独立识别老师手写的完整红圈、红叉、删除线、下划线、批注或其他明确错误标记。

要求：
1. 不接收也不得推测任何外部候选坐标；逐个观察原图中的完整几何形状。
2. circle 的 bbox 必须覆盖完整闭合或近似闭合红圈；cross 的 bbox 必须覆盖两条相交笔画。不得把圆弧、叉的一条笔画或孤立红线作为独立标记。
3. 此阶段只输出独立 primitive，不配对圈叉，不识别题目内容。
4. 不要把印刷红色方格、页眉线、装饰色或单独红色对勾当作错误标记。
5. 教师文字批注必须标为 annotation，不得因为包含相交笔画而标为 cross；此类批注仍作为 primitive 输出供后续归属审计。
6. bbox 使用整页归一化 [left, top, right, bottom]；mark_id 从 0 开始连续且唯一。
7. 只返回严格 JSON：{"error_marks":[{"mark_id":0,"mark_type":"circle","bbox":[0.1,0.2,0.3,0.4],"cross_bbox":null,"circle_bbox":null,"confidence":0.95}]}。
"""


CV_CROSS_VERIFICATION_PROMPT = """你是小学作业红叉候选核验器。图片是候选核验拼图：左侧为原始整页作业预览，右侧为标有 X0、X1... 的本地 CV 候选局部放大图。

本地 CV 以召回优先，候选中可能包含印刷文字、红色方格、页码、边缘或老师批注造成的误报，也可能漏掉真实红叉。

要求：
1. 逐个独立处置输入 candidate_id；每个 candidate_id 必须且只能返回一个 verdict，不得合并、关联或重排不同候选。
2. disposition=confirmed 仅表示该候选局部能看到两条红色相交笔画的老师判错红叉；rejected 表示明确不是红叉；无法确定时必须返回 uncertain。
3. 印刷文字、红色方格、页码、装饰线、图片边缘、红圈弧线、单条红线及教师文字批注不得确认为红叉。
4. 此请求只核验编号候选，不负责扫描整页漏检。
5. 不得返回或重画 bbox；后续锚点坐标只使用本地 CV 原始候选框，并由本地确定性几何逻辑去重。
6. 此阶段不识别题目内容，不匹配红圈，不决定错题区域。
7. 只返回严格 JSON，不要解释或 Markdown。

返回格式：{"verdicts":[{"candidate_id":0,"disposition":"confirmed","confidence":0.95},{"candidate_id":1,"disposition":"rejected","confidence":0.9},{"candidate_id":2,"disposition":"uncertain","confidence":0.5}]}。

输入 CV candidates：__CANDIDATES__
"""


INDEPENDENT_CROSS_SCAN_PROMPT = """你是小学作业红叉独立召回器。只查看当前原始整页作业图片，不参考任何本地 CV 候选或其他模型结果。

目标：召回整页所有由老师红笔画出的、表示学生作答错误的完整红叉 X。

要求：
1. 红叉必须能看到两条红色相交笔画；bbox 覆盖完整红叉。
2. 不得把红圈弧线、单条红线、箭头、教师文字批注、红色方格、印刷文字、页码、装饰线或图片边缘当成红叉。
3. 同一个红叉只能返回一次；不识别题目内容，不匹配错题区域。
4. bbox 使用原始整页归一化 [left, top, right, bottom]。
5. 只返回严格 JSON，不要解释或 Markdown。

返回格式：{"crosses":[{"bbox":[0.1,0.2,0.3,0.4],"confidence":0.95}]}。
"""


FALLBACK_CROSS_VERIFICATION_PROMPT = """你是小学作业独立扫描红叉复核器。图片左侧是原始整页预览，右侧是标有 X0、X1... 的独立扫描候选局部放大图。这些候选没有本地 CV 几何支持，误报风险较高。

要求：
1. 每个 candidate_id 必须且只能返回一个独立 verdict，不得合并候选。
2. 只有局部图中清楚看到两条老师红笔相交笔画，并且它表示学生作答错误时，才能返回 disposition=confirmed。
3. 红圈弧线、箭头、教师文字、印刷方格、页码、单条红线或无法确认的形状不得确认为红叉；无法确定时返回 uncertain。
4. 不返回 bbox，不识别题目内容。只返回严格 JSON，不要解释或 Markdown。

返回格式：{"verdicts":[{"candidate_id":0,"disposition":"confirmed","confidence":0.95},{"candidate_id":1,"disposition":"rejected","confidence":0.9}]}。

输入 fallback candidates：__CANDIDATES__
"""


CROSS_ANCHORED_QUESTION_PROMPT = """你是小学作业红叉锚定错题区域定位器。图片可能是原始整页作业，也可能是单个可疑红叉附近的局部放大图；图片上叠加了蓝色 C0、C1... 编号框，输入 JSON 给出每个 cross_id 在当前输入图片坐标系中的位置和来源风险。

本阶段唯一目标：逐个判断红叉候选能否关联到一个明确的最小独立作答单元，并只返回完整 question_bbox。不要识别或返回题目文字、学生答案、正确答案、题型、标签、难度、红圈或教师批注。

要求：
1. 每个输入 cross_id 必须且只能返回一次，不得新增、删除、合并或重排。输入可能含误报；若看不到明确老师判错红叉，或无法唯一关联作答单元，必须返回 matched=false，并简要填写 unmatched_reason。
2. matched=true 时，question_bbox 覆盖与该红叉直接相关的完整最小作答单元：印刷提示、学生实际作答以及相关批改痕迹；红叉可以位于框内，也可以紧邻该作答单元，不得因红叉画在作答框上方、右侧或外沿而返回 unmatched；不得吞入相邻兄弟小题或整页区域。
3. question_bbox 不得直接复制红叉框。红叉框只标出批改痕迹，question_bbox 必须扩展到完整印刷提示和学生作答；如果无法看清完整边界，返回 matched=false。
4. source=llm_fallback 表示低可信独立扫描结果，不得默认其为真实红叉，必须根据图片重新核验。
5. matched=false 时 question_bbox 必须为 null；matched=true 时 unmatched_reason 应为 null。
6. question_bbox 使用当前输入图片的归一化 [left, top, right, bottom]。若 anchor 的 image_scope=local_retry_crop，必须按当前局部放大图返回坐标；蓝色编号框只是提示，不是图片原有内容。
7. 只返回严格 JSON，不要解释或 Markdown。

返回格式：{"items":[{"cross_id":0,"matched":true,"question_bbox":[0.1,0.2,0.4,0.5],"unmatched_reason":null,"confidence":0.95},{"cross_id":1,"matched":false,"question_bbox":null,"unmatched_reason":"不是明确红叉","confidence":0.8}]}。

输入 cross anchors：__CROSSES__
"""


class StableMarkEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event_id: int = Field(ge=0)
    primitive_ids: list[int] = Field(min_length=1)
    event_type: Literal[
        "cross_circle",
        "circle",
        "cross",
        "deletion",
        "underline",
        "annotation",
        "mixed",
    ]
    bbox: list[float]
    cross_bbox: list[float] | None = None
    circle_bbox: list[float] | None = None
    confidence: float = Field(ge=0, le=1)

    @field_validator("event_type", mode="before")
    @classmethod
    def normalize_event_type_alias(cls, value):
        return "cross" if value == "cross_only" else value

    @field_validator("cross_bbox", "circle_bbox", mode="before")
    @classmethod
    def normalize_empty_component_bbox(cls, value):
        return None if value == [0, 0, 0, 0] else value

    @field_validator("bbox", "cross_bbox", "circle_bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        return validate_normalized_bbox(value) if value is not None else None

    @model_validator(mode="after")
    def component_boxes_must_match_event_type(self):
        if self.event_type == "cross_circle" and not (
            self.cross_bbox and self.circle_bbox
        ):
            raise ValueError("cross_circle requires cross_bbox and circle_bbox")
        if self.event_type == "circle" and self.circle_bbox is None:
            raise ValueError("circle requires circle_bbox")
        if self.event_type == "cross" and self.cross_bbox is None:
            raise ValueError("cross requires cross_bbox")
        return self


class StableEventResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    events: list[StableMarkEvent] = Field(default_factory=list)
    unassigned_primitive_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def assignments_must_be_unique(self):
        event_ids = [event.event_id for event in self.events]
        if event_ids != list(range(len(event_ids))):
            raise ValueError("stable event ids must be unique and sequential")
        assigned = [
            primitive_id
            for event in self.events
            for primitive_id in event.primitive_ids
        ]
        duplicates = sorted(
            primitive_id
            for primitive_id in set(assigned)
            if assigned.count(primitive_id) > 1
        )
        if duplicates:
            raise ValueError(f"primitive ids assigned more than once: {duplicates}")
        if len(self.unassigned_primitive_ids) != len(
            set(self.unassigned_primitive_ids)
        ):
            raise ValueError("unassigned primitive ids must be unique")
        overlap = sorted(set(assigned) & set(self.unassigned_primitive_ids))
        if overlap:
            raise ValueError(f"primitive ids both assigned and unassigned: {overlap}")
        return self


StableEventResult.model_rebuild(_types_namespace=globals())


class CandidateCrossVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_id: int = Field(ge=0)
    disposition: Literal["confirmed", "rejected", "uncertain"]
    confidence: float = Field(ge=0, le=1)


class FallbackCross(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    bbox: list[float]
    confidence: float = Field(ge=0, le=1)

    @field_validator("bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        return validate_normalized_bbox(value)


class CrossCandidateVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    verdicts: list[CandidateCrossVerdict] = Field(default_factory=list)


class IndependentCrossScanResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    crosses: list[FallbackCross] = Field(default_factory=list)


class CrossAnchoredQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cross_id: int = Field(ge=0)
    matched: bool
    question_bbox: list[float] | None = None
    unmatched_reason: str | None = None
    confidence: float = Field(ge=0, le=1)

    @field_validator("question_bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        return validate_normalized_bbox(value) if value is not None else None

    @model_validator(mode="after")
    def matched_fields_must_be_consistent(self):
        if self.matched:
            if self.question_bbox is None:
                raise ValueError("matched cross anchor requires question_bbox")
            if self.unmatched_reason is not None:
                raise ValueError("matched cross anchor cannot have unmatched_reason")
        elif self.question_bbox is not None:
            raise ValueError("unmatched cross anchor cannot have question_bbox")
        elif not self.unmatched_reason or not self.unmatched_reason.strip():
            raise ValueError("unmatched cross anchor requires reason")
        return self


class CrossAnchoredQuestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: list[CrossAnchoredQuestion] = Field(default_factory=list)


CrossCandidateVerificationResult.model_rebuild(_types_namespace=globals())
IndependentCrossScanResult.model_rebuild(_types_namespace=globals())
CrossAnchoredQuestionResult.model_rebuild(_types_namespace=globals())


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pixel_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(bbox[0] * width),
        round(bbox[1] * height),
        round(bbox[2] * width),
        round(bbox[3] * height),
    )


def _draw_boxes(
    image_path: Path,
    output_path: Path,
    layers: list[tuple[str, list[dict], str]],
) -> None:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    for prefix, entries, color in layers:
        for index, entry in enumerate(entries):
            bbox = entry.get("bbox")
            if not bbox:
                continue
            pixel_bbox = _pixel_bbox(bbox, image.width, image.height)
            draw.rectangle(pixel_bbox, outline=color, width=max(2, image.width // 600))
            draw.text(
                (pixel_bbox[0] + 2, pixel_bbox[1] + 2),
                f"{prefix}{entry.get('mark_id', entry.get('region_id', index))}",
                fill=color,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=92)


def write_cross_candidate_montage(
    image_path: Path,
    output_path: Path,
    candidates: list[dict],
    *,
    full_page_max_edge: int,
    tile_edge: int,
    columns: int,
    crop_padding_ratio: float,
) -> dict:
    with Image.open(image_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    full_page = original.copy()
    full_page.thumbnail(
        (full_page_max_edge, full_page_max_edge),
        Image.Resampling.LANCZOS,
    )
    label_height = 28
    gap = 12
    tile_columns = min(columns, max(1, len(candidates)))
    tile_rows = max(1, math.ceil(len(candidates) / tile_columns))
    tiles_width = tile_columns * tile_edge
    tiles_height = tile_rows * (tile_edge + label_height)
    montage = Image.new(
        "RGB",
        (full_page.width + gap + tiles_width, max(full_page.height, tiles_height)),
        "white",
    )
    montage.paste(full_page, (0, 0))
    draw = ImageDraw.Draw(montage)
    for tile_index, candidate in enumerate(candidates):
        column = tile_index % tile_columns
        row = tile_index // tile_columns
        tile_x = full_page.width + gap + column * tile_edge
        tile_y = row * (tile_edge + label_height)
        bbox = candidate["bbox"]
        crop_bbox = [
            max(0.0, bbox[0] - crop_padding_ratio),
            max(0.0, bbox[1] - crop_padding_ratio),
            min(1.0, bbox[2] + crop_padding_ratio),
            min(1.0, bbox[3] + crop_padding_ratio),
        ]
        pixel_bbox = _pixel_bbox(crop_bbox, original.width, original.height)
        crop = original.crop(pixel_bbox)
        enlarged = ImageOps.contain(
            crop,
            (tile_edge - 4, tile_edge - 4),
            Image.Resampling.LANCZOS,
        )
        paste_x = tile_x + (tile_edge - enlarged.width) // 2
        paste_y = tile_y + (tile_edge - enlarged.height) // 2
        montage.paste(enlarged, (paste_x, paste_y))
        draw.rectangle(
            (tile_x, tile_y, tile_x + tile_edge - 1, tile_y + tile_edge - 1),
            outline="blue",
            width=3,
        )
        draw.text(
            (tile_x + 4, tile_y + tile_edge + 4),
            f"X{candidate['candidate_id']} bbox={bbox}",
            fill="blue",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path, format="JPEG", quality=92)
    return {
        "candidate_count": len(candidates),
        "tile_candidate_ids": [
            candidate["candidate_id"] for candidate in candidates
        ],
    }


def _red_mask_for_image(image_path: Path, max_edge: int) -> np.ndarray:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.int16)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    return (
        (red >= 120)
        & (red - green >= 45)
        & (red - blue >= 45)
        & (red >= green * 1.35)
        & (red >= blue * 1.35)
    )


def _cross_arm_offsets(
    inner_radius: int,
    outer_radius: int,
    diagonal_band_ratio: float,
) -> dict[tuple[int, int], np.ndarray]:
    offsets: dict[tuple[int, int], list[tuple[int, int]]] = {
        (-1, -1): [],
        (-1, 1): [],
        (1, -1): [],
        (1, 1): [],
    }
    for delta_y in range(-outer_radius, outer_radius + 1):
        for delta_x in range(-outer_radius, outer_radius + 1):
            distance = max(abs(delta_x), abs(delta_y))
            if not inner_radius <= distance <= outer_radius:
                continue
            diagonal_gap = abs(abs(delta_x) - abs(delta_y))
            if diagonal_gap > max(1, round(distance * diagonal_band_ratio)):
                continue
            quadrant = (
                -1 if delta_y < 0 else 1,
                -1 if delta_x < 0 else 1,
            )
            offsets[quadrant].append((delta_y, delta_x))
    return {
        quadrant: np.asarray(values, dtype=np.int32)
        for quadrant, values in offsets.items()
    }


def detect_red_cross_candidates(image_path: Path, config: dict) -> dict:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        analysis_max_edge = int(config["analysis_max_edge"])
        if max(image.size) > analysis_max_edge:
            image.thumbnail(
                (analysis_max_edge, analysis_max_edge),
                Image.Resampling.LANCZOS,
            )

    pixels = np.asarray(image, dtype=np.int16)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    red_excess = red - np.maximum(green, blue)
    red_mask = (red >= int(config["red_min_channel"])) & (
        red_excess >= int(config["red_min_excess"])
    )

    height, width = red_mask.shape
    scale = max(width, height)
    inner_radius = max(1, round(scale * float(config["arm_inner_radius_ratio"])))
    outer_radius = max(
        inner_radius + 1,
        round(scale * float(config["arm_outer_radius_ratio"])),
    )
    center_radius = max(1, round(scale * float(config["center_radius_ratio"])))
    merge_radius = max(
        1,
        round(scale * float(config["candidate_merge_radius_ratio"])),
    )
    padding = max(0, round(scale * float(config["bbox_padding_ratio"])))
    axis_offsets = range(-outer_radius, outer_radius + 1)
    horizontal_support = np.zeros(red_mask.shape, dtype=np.uint16)
    vertical_support = np.zeros(red_mask.shape, dtype=np.uint16)
    for offset in axis_offsets:
        if offset < 0:
            horizontal_support[:, :offset] += red_mask[:, -offset:]
            vertical_support[:offset, :] += red_mask[-offset:, :]
        elif offset > 0:
            horizontal_support[:, offset:] += red_mask[:, :-offset]
            vertical_support[offset:, :] += red_mask[:-offset, :]
        else:
            horizontal_support += red_mask
            vertical_support += red_mask
    axis_sample_count = outer_radius * 2 + 1
    axis_line_min_density = float(config["axis_line_min_density"])
    geometry_mask = red_mask & (
        (horizontal_support / axis_sample_count < axis_line_min_density)
        & (vertical_support / axis_sample_count < axis_line_min_density)
    )
    arm_offsets = _cross_arm_offsets(
        inner_radius,
        outer_radius,
        float(config["diagonal_band_ratio"]),
    )
    arm_min_density = float(config["arm_min_density"])
    center_min_density = float(config["center_min_density"])

    scored_centers = []
    for center_y, center_x in np.argwhere(red_mask):
        left = max(0, center_x - center_radius)
        top = max(0, center_y - center_radius)
        right = min(width, center_x + center_radius + 1)
        bottom = min(height, center_y + center_radius + 1)
        center_density = float(red_mask[top:bottom, left:right].mean())
        if center_density < center_min_density:
            continue

        arm_densities = []
        for offsets in arm_offsets.values():
            sample_y = center_y + offsets[:, 0]
            sample_x = center_x + offsets[:, 1]
            valid = (
                (sample_x >= 0)
                & (sample_x < width)
                & (sample_y >= 0)
                & (sample_y < height)
            )
            density = (
                float(geometry_mask[sample_y[valid], sample_x[valid]].mean())
                if valid.any()
                else 0.0
            )
            arm_densities.append(density)
        minimum_density = min(arm_densities)
        if minimum_density >= arm_min_density:
            scored_centers.append(
                {
                    "x": int(center_x),
                    "y": int(center_y),
                    "min_arm_density": minimum_density,
                    "arm_densities": arm_densities,
                    "center_density": center_density,
                }
            )

    clusters: list[list[dict]] = []
    for center in scored_centers:
        matching_cluster = next(
            (
                cluster
                for cluster in clusters
                if min(
                    math.hypot(center["x"] - item["x"], center["y"] - item["y"])
                    for item in cluster
                )
                <= merge_radius
            ),
            None,
        )
        if matching_cluster is None:
            clusters.append([center])
        else:
            matching_cluster.append(center)

    candidates = []
    candidate_center_mask = np.zeros_like(red_mask)
    for cluster in clusters:
        best = max(
            cluster,
            key=lambda item: (
                item["min_arm_density"],
                item["center_density"],
                -item["y"],
                -item["x"],
            ),
        )
        for item in cluster:
            candidate_center_mask[item["y"], item["x"]] = True
        radius = outer_radius + padding
        left = max(0, best["x"] - radius)
        top = max(0, best["y"] - radius)
        right = min(width, best["x"] + radius + 1)
        bottom = min(height, best["y"] + radius + 1)
        candidates.append(
            {
                "candidate_id": 0,
                "bbox": [
                    round(left / width, 6),
                    round(top / height, 6),
                    round(right / width, 6),
                    round(bottom / height, 6),
                ],
                "center": [
                    round(best["x"] / width, 6),
                    round(best["y"] / height, 6),
                ],
                "min_arm_density": round(best["min_arm_density"], 6),
                "arm_densities": [round(value, 6) for value in best["arm_densities"]],
                "center_density": round(best["center_density"], 6),
            }
        )

    candidates.sort(key=lambda item: (item["center"][1], item["center"][0]))
    for candidate_id, candidate in enumerate(candidates):
        candidate["candidate_id"] = candidate_id
    return {
        "analysis_width": width,
        "analysis_height": height,
        "red_pixel_count": int(red_mask.sum()),
        "candidates": candidates,
        "red_mask": red_mask,
        "geometry_mask": geometry_mask,
        "candidate_center_mask": candidate_center_mask,
    }


def compare_cross_candidates_to_truth(
    candidates: list[dict],
    truth_regions: list[dict],
    *,
    margin_ratio: float = 0.0,
) -> dict:
    assignments = []
    matched_truth_ids = set()
    false_candidate_ids = []
    for candidate in candidates:
        center_x, center_y = candidate["center"]
        containing = [
            truth
            for truth in truth_regions
            if truth["source_bbox_normalized"][0]
            - margin_ratio
            <= center_x
            <= truth["source_bbox_normalized"][2] + margin_ratio
            and truth["source_bbox_normalized"][1]
            - margin_ratio
            <= center_y
            <= truth["source_bbox_normalized"][3] + margin_ratio
        ]
        if containing:
            truth = min(
                containing,
                key=lambda item: (
                    (item["source_bbox_normalized"][2] - item["source_bbox_normalized"][0])
                    * (item["source_bbox_normalized"][3] - item["source_bbox_normalized"][1]),
                    item["truth_id"],
                ),
            )
            truth_id = truth["truth_id"]
            matched_truth_ids.add(truth_id)
        else:
            truth_id = None
            false_candidate_ids.append(candidate["candidate_id"])
        assignments.append(
            {"candidate_id": candidate["candidate_id"], "truth_id": truth_id}
        )

    ordered_truth_ids = [truth["truth_id"] for truth in truth_regions]
    matched = [truth_id for truth_id in ordered_truth_ids if truth_id in matched_truth_ids]
    missed = [truth_id for truth_id in ordered_truth_ids if truth_id not in matched_truth_ids]
    truth_count = len(truth_regions)
    return {
        "truth_count": truth_count,
        "candidate_count": len(candidates),
        "matched_truth_count": len(matched),
        "truth_recall": round(len(matched) / truth_count, 6) if truth_count else None,
        "matched_truth_ids": matched,
        "missed_truth_ids": missed,
        "false_candidate_ids": false_candidate_ids,
        "assignments": assignments,
    }


def write_cross_cv_artifacts(
    image_path: Path,
    case_dir: Path,
    config: dict,
    truth_regions: list[dict] | None = None,
) -> dict:
    experiment_dir = case_dir / "cv-cross-experiment"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    _write_json(experiment_dir / "effective-config.json", config)
    detected = detect_red_cross_candidates(image_path, config)
    candidates = detected["candidates"]
    Image.fromarray((detected["red_mask"] * 255).astype(np.uint8), mode="L").save(
        experiment_dir / "red-mask.png"
    )
    Image.fromarray(
        (detected["geometry_mask"] * 255).astype(np.uint8), mode="L"
    ).save(experiment_dir / "geometry-mask.png")
    Image.fromarray(
        (detected["candidate_center_mask"] * 255).astype(np.uint8), mode="L"
    ).save(experiment_dir / "candidate-centers.png")
    _write_json(experiment_dir / "candidates.json", candidates)
    candidate_entries = [
        {"mark_id": candidate["candidate_id"], "bbox": candidate["bbox"]}
        for candidate in candidates
    ]
    _draw_boxes(
        image_path,
        experiment_dir / "candidates-overlay.jpg",
        [("X", candidate_entries, "blue")],
    )

    truth_comparison = None
    if truth_regions is not None:
        truth_comparison = compare_cross_candidates_to_truth(
            candidates,
            truth_regions,
            margin_ratio=float(config.get("truth_match_margin_ratio", 0.0)),
        )
        _write_json(
            experiment_dir / "truth-comparison.json",
            truth_comparison,
        )
        truth_entries = [
            {
                "mark_id": truth["truth_id"],
                "bbox": truth["source_bbox_normalized"],
            }
            for truth in truth_regions
        ]
        _draw_boxes(
            image_path,
            experiment_dir / "truth-candidates-overlay.jpg",
            [("", truth_entries, "green"), ("X", candidate_entries, "blue")],
        )

    return {
        "analysis_width": detected["analysis_width"],
        "analysis_height": detected["analysis_height"],
        "red_pixel_count": detected["red_pixel_count"],
        "candidate_count": len(candidates),
        "truth_comparison": truth_comparison,
    }


def write_cv_artifacts(
    image_path: Path,
    output_dir: Path,
    *,
    max_edge: int,
    min_component_pixels: int,
    max_component_area_ratio: float,
    max_thinness_ratio: float,
    group_max_gap_ratio: float,
    group_max_area_ratio: float,
) -> dict:
    scan = scan_red_mark_regions(
        str(image_path),
        max_edge=max_edge,
        min_component_pixels=min_component_pixels,
        max_component_area_ratio=max_component_area_ratio,
        max_thinness_ratio=max_thinness_ratio,
    )
    groups, grouping = group_red_evidence_regions(
        scan.regions,
        max_gap_ratio=group_max_gap_ratio,
        max_group_area_ratio=group_max_area_ratio,
    )
    components = [
        {"region_id": index, **region.model_dump(mode="json")}
        for index, region in enumerate(scan.regions)
    ]
    grouped = [
        {"region_id": index, **region.model_dump(mode="json")}
        for index, region in enumerate(groups)
    ]
    payload = {
        "scan": scan.model_dump(mode="json", exclude={"regions"}),
        "components": components,
        "groups": grouped,
        **grouping,
    }
    _write_json(output_dir / "cv" / "evidence.json", payload)
    red_mask = _red_mask_for_image(image_path, max_edge)
    Image.fromarray((red_mask * 255).astype(np.uint8), mode="L").save(
        output_dir / "cv" / "red-mask.png"
    )
    _draw_boxes(
        image_path,
        output_dir / "cv" / "components-and-groups.jpg",
        [("C", components, "red"), ("G", grouped, "blue")],
    )
    return {**payload, "scan_result": scan, "red_mask": red_mask}


def validate_stable_event_assignment(
    result: StableEventResult,
    *,
    primitive_count: int,
) -> dict:
    assigned = [
        primitive_id
        for event in result.events
        for primitive_id in event.primitive_ids
    ]
    duplicates = sorted(
        primitive_id
        for primitive_id in set(assigned)
        if assigned.count(primitive_id) > 1
    )
    if duplicates:
        raise ValueError(f"primitive ids assigned more than once: {duplicates}")

    known = set(range(primitive_count))
    assigned_set = set(assigned)
    unassigned = result.unassigned_primitive_ids
    unassigned_set = set(unassigned)
    if len(unassigned) != len(unassigned_set):
        raise ValueError("unassigned primitive ids must be unique")
    unknown = sorted((assigned_set | unassigned_set) - known)
    if unknown:
        raise ValueError(f"unknown primitive ids: {unknown}")
    overlap = sorted(assigned_set & unassigned_set)
    if overlap:
        raise ValueError(f"primitive ids both assigned and unassigned: {overlap}")
    missing = sorted(known - assigned_set - unassigned_set)
    if missing:
        raise ValueError(f"primitive ids missing from assignment: {missing}")
    return {
        "duplicate_primitive_ids": duplicates,
        "assigned_primitive_ids": sorted(assigned_set),
        "unassigned_primitive_ids": sorted(unassigned_set),
    }


def audit_stable_event_primitive_membership(
    result: StableEventResult,
    primitives: list[ErrorMark],
) -> list[dict]:
    primitive_types = {
        primitive.mark_id: primitive.mark_type for primitive in primitives
    }
    violations = []
    for event in result.events:
        actual_types = sorted(
            primitive_types.get(primitive_id, "unknown")
            for primitive_id in event.primitive_ids
        )
        expected_types = (
            ["circle", "cross"]
            if event.event_type == "cross_circle"
            else [event.event_type]
        )
        if actual_types != expected_types:
            violations.append(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "primitive_ids": list(event.primitive_ids),
                    "primitive_types": actual_types,
                    "expected_primitive_types": expected_types,
                }
            )
    return violations


def _bbox_intersection_area(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_iou(first: list[float], second: list[float]) -> float:
    intersection = _bbox_intersection_area(first, second)
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union else 0.0


def audit_cross_candidate_dispositions(
    result: CrossCandidateVerificationResult,
    candidate_ids: list[int],
) -> dict:
    dispositions = [verdict.candidate_id for verdict in result.verdicts]
    input_ids = sorted(candidate_ids)
    input_set = set(input_ids)
    disposition_set = set(dispositions)
    duplicate_ids = sorted(
        candidate_id
        for candidate_id in disposition_set
        if dispositions.count(candidate_id) > 1
    )
    missing_ids = sorted(input_set - disposition_set)
    unknown_ids = sorted(disposition_set - input_set)
    return {
        "valid": not (missing_ids or unknown_ids or duplicate_ids),
        "input_candidate_ids": input_ids,
        "missing_candidate_ids": missing_ids,
        "unknown_candidate_ids": unknown_ids,
        "duplicate_candidate_ids": duplicate_ids,
        "policy": "Every CV candidate must be dispositioned exactly once.",
    }


def aggregate_cross_candidate_verifications(
    runs: list[CrossCandidateVerificationResult],
) -> tuple[CrossCandidateVerificationResult, dict]:
    candidate_ids = sorted(
        {verdict.candidate_id for run in runs for verdict in run.verdicts}
    )
    priority = ("confirmed", "uncertain", "rejected")
    aggregated_verdicts = []
    candidate_audits = []
    unstable_candidate_ids = []
    for candidate_id in candidate_ids:
        verdicts = [
            next(
                verdict
                for verdict in run.verdicts
                if verdict.candidate_id == candidate_id
            )
            for run in runs
        ]
        dispositions = [verdict.disposition for verdict in verdicts]
        aggregated_disposition = next(
            disposition for disposition in priority if disposition in dispositions
        )
        matching_confidences = [
            verdict.confidence
            for verdict in verdicts
            if verdict.disposition == aggregated_disposition
        ]
        aggregated_verdicts.append(
            CandidateCrossVerdict(
                candidate_id=candidate_id,
                disposition=aggregated_disposition,
                confidence=max(matching_confidences),
            )
        )
        most_common_count = max(dispositions.count(item) for item in set(dispositions))
        if len(set(dispositions)) > 1:
            unstable_candidate_ids.append(candidate_id)
        candidate_audits.append(
            {
                "candidate_id": candidate_id,
                "dispositions": dispositions,
                "aggregated_disposition": aggregated_disposition,
                "agreement_ratio": round(most_common_count / len(runs), 6),
            }
        )
    return (
        CrossCandidateVerificationResult(verdicts=aggregated_verdicts),
        {
            "run_count": len(runs),
            "unstable_candidate_ids": unstable_candidate_ids,
            "candidates": candidate_audits,
            "policy": "confirmed wins, then uncertain; rejected requires every run to reject",
        },
    )


def _bbox_center_distance(first: list[float], second: list[float]) -> float:
    first_x = (first[0] + first[2]) / 2
    first_y = (first[1] + first[3]) / 2
    second_x = (second[0] + second[2]) / 2
    second_y = (second[1] + second[3]) / 2
    return ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5


def _bbox_gap_distance(first: list[float], second: list[float]) -> float:
    horizontal_gap = max(first[0] - second[2], second[0] - first[2], 0.0)
    vertical_gap = max(first[1] - second[3], second[1] - first[3], 0.0)
    return (horizontal_gap**2 + vertical_gap**2) ** 0.5


def _bbox_union(first: list[float], second: list[float]) -> list[float]:
    return [
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    ]


def _dedupe_cv_anchors(anchors: list[dict], config: dict) -> list[dict]:
    clusters = []
    for anchor in sorted(
        anchors,
        key=lambda item: item["source_candidate_ids"][0],
    ):
        matching_cluster = next(
            (
                cluster
                for cluster in clusters
                if all(
                    _bbox_iou(member["bbox"], anchor["bbox"])
                    >= config["cross_anchor_cv_dedupe_iou_threshold"]
                    or _bbox_center_distance(member["bbox"], anchor["bbox"])
                    <= config["cross_anchor_cv_dedupe_center_distance_ratio"]
                    for member in cluster
                )
            ),
            None,
        )
        if matching_cluster is None:
            clusters.append([anchor])
        else:
            matching_cluster.append(anchor)

    deduped = []
    source_priority = {
        "cv_confirmed": 0,
        "cv_uncertain": 1,
        "cv_high_score_retained": 2,
        "cv_rejected_retained": 3,
    }
    for cluster in clusters:
        representative = min(
            cluster,
            key=lambda item: (
                source_priority[item["source"]],
                -float(item["confidence"] or 0),
                item["source_candidate_ids"][0],
            ),
        )
        bbox = list(cluster[0]["bbox"])
        for member in cluster[1:]:
            bbox = _bbox_union(bbox, member["bbox"])
        deduped.append(
            {
                "source": representative["source"],
                "source_candidate_ids": sorted(
                    candidate_id
                    for member in cluster
                    for candidate_id in member["source_candidate_ids"]
                ),
                "bbox": bbox,
                "confidence": representative["confidence"],
                "independent_scan_supported": False,
                "merge_reason": "local_geometry" if len(cluster) > 1 else None,
            }
        )
    return deduped


def build_cross_anchors(
    verification: CrossCandidateVerificationResult,
    independent_scan: IndependentCrossScanResult,
    candidates: list[dict],
    config: dict,
) -> list[dict]:
    candidate_by_id = {
        candidate["candidate_id"]: candidate for candidate in candidates
    }
    candidate_anchors = []
    for verdict in verification.verdicts:
        candidate = candidate_by_id[verdict.candidate_id]
        source = None
        if verdict.disposition == "confirmed":
            source = "cv_confirmed"
        elif (
            verdict.disposition == "uncertain"
            and config["cross_anchor_retain_uncertain_candidates"]
        ):
            source = "cv_uncertain"
        elif (
            verdict.disposition == "rejected"
            and candidate["min_arm_density"]
            >= config["cross_anchor_high_cv_min_arm_density"]
            and candidate["center_density"]
            >= config["cross_anchor_high_cv_min_center_density"]
        ):
            source = "cv_high_score_retained"
        elif (
            verdict.disposition == "rejected"
            and config.get("cross_anchor_retain_rejected_candidates", False)
        ):
            source = "cv_rejected_retained"
        if source is not None:
            candidate_anchors.append(
                {
                    "source": source,
                    "source_candidate_ids": [verdict.candidate_id],
                    "bbox": list(candidate["bbox"]),
                    "confidence": verdict.confidence,
                }
            )
    anchors = _dedupe_cv_anchors(candidate_anchors, config)
    for cross in independent_scan.crosses:
        matching_anchor = next(
            (
                anchor
                for anchor in anchors
                if _bbox_iou(anchor["bbox"], cross.bbox)
                >= config["cross_anchor_fallback_merge_iou_threshold"]
                or _bbox_center_distance(anchor["bbox"], cross.bbox)
                <= config["cross_anchor_fallback_merge_center_distance_ratio"]
            ),
            None,
        )
        if matching_anchor is not None:
            matching_anchor["independent_scan_supported"] = True
        else:
            anchors.append(
                {
                    "source": "llm_fallback",
                    "source_candidate_ids": [],
                    "bbox": list(cross.bbox),
                    "confidence": cross.confidence,
                    "independent_scan_supported": True,
                    "merge_reason": None,
                }
            )
    anchors.sort(
        key=lambda item: (
            (item["bbox"][1] + item["bbox"][3]) / 2,
            (item["bbox"][0] + item["bbox"][2]) / 2,
            item["source"],
        )
    )
    for cross_id, anchor in enumerate(anchors):
        anchor["cross_id"] = cross_id
    return [
        {
            "cross_id": anchor["cross_id"],
            "source": anchor["source"],
            "source_candidate_ids": anchor["source_candidate_ids"],
            "bbox": anchor["bbox"],
            "confidence": anchor["confidence"],
            "independent_scan_supported": anchor["independent_scan_supported"],
            "merge_reason": anchor["merge_reason"],
        }
        for anchor in anchors
    ]


def audit_cross_anchor_assignments(
    result: CrossAnchoredQuestionResult,
    cross_ids: list[int],
) -> dict:
    returned_ids = [item.cross_id for item in result.items]
    expected_ids = sorted(cross_ids)
    expected_set = set(expected_ids)
    returned_set = set(returned_ids)
    duplicate_ids = sorted(
        cross_id for cross_id in returned_set if returned_ids.count(cross_id) > 1
    )
    missing_ids = sorted(expected_set - returned_set)
    unknown_ids = sorted(returned_set - expected_set)
    return {
        "valid": not (missing_ids or unknown_ids or duplicate_ids),
        "input_cross_ids": expected_ids,
        "missing_cross_ids": missing_ids,
        "unknown_cross_ids": unknown_ids,
        "duplicate_cross_ids": duplicate_ids,
        "policy": "Every confirmed cross must be returned exactly once.",
    }


def audit_anchored_question_geometry(
    result: CrossAnchoredQuestionResult,
    anchors: list[dict],
    *,
    max_area_ratio: float,
    max_gap_ratio: float,
) -> dict:
    anchor_by_id = {anchor["cross_id"]: anchor for anchor in anchors}
    violations = []
    for item in result.items:
        if not item.matched or item.question_bbox is None:
            continue
        reasons = []
        area_ratio = _bbox_area(item.question_bbox)
        anchor = anchor_by_id.get(item.cross_id)
        if anchor is None:
            reasons.append("unknown_cross_id")
        else:
            cross_bbox = anchor["bbox"]
            if _bbox_gap_distance(item.question_bbox, cross_bbox) > max_gap_ratio:
                reasons.append("question_bbox_not_near_cross")
        if area_ratio > max_area_ratio:
            reasons.append("question_bbox_exceeds_max_area_ratio")
        if reasons:
            violations.append(
                {
                    "cross_id": item.cross_id,
                    "reasons": reasons,
                    "question_area_ratio": round(area_ratio, 6),
                }
            )
    return {
        "valid": not violations,
        "max_area_ratio": max_area_ratio,
        "max_gap_ratio": max_gap_ratio,
        "violations": violations,
        "policy": "Geometry audit records violations and never rewrites LLM output.",
    }


def _point_bbox_gap_distance(point: tuple[float, float], bbox: list[float]) -> float:
    x, y = point
    dx = max(bbox[0] - x, 0.0, x - bbox[2])
    dy = max(bbox[1] - y, 0.0, y - bbox[3])
    return math.hypot(dx, dy)


def select_llm2_retry_anchors(
    first_pass: CrossAnchoredQuestionResult,
    anchors: list[dict],
    geometry_audit: dict,
    *,
    min_center_gap_ratio: float,
) -> dict:
    items_by_id = {item.cross_id: item for item in first_pass.items}
    geometry_reasons = {
        violation["cross_id"]: list(violation["reasons"])
        for violation in geometry_audit["violations"]
    }
    high_evidence_sources = {
        "cv_confirmed",
        "cv_uncertain",
        "cv_high_score_retained",
        "llm_fallback",
    }
    selected = []
    triggers = []
    for anchor in anchors:
        cross_id = anchor["cross_id"]
        item = items_by_id[cross_id]
        reasons = []
        center_gap = None
        if item.matched and item.question_bbox is not None:
            cross_bbox = anchor["bbox"]
            cross_center = (
                (cross_bbox[0] + cross_bbox[2]) / 2,
                (cross_bbox[1] + cross_bbox[3]) / 2,
            )
            center_gap = _point_bbox_gap_distance(cross_center, item.question_bbox)
            if center_gap > min_center_gap_ratio:
                reasons.append("cross_center_outside_question_bbox")
            reasons.extend(geometry_reasons.get(cross_id, []))
        elif (
            anchor.get("source") in high_evidence_sources
            or anchor.get("independent_scan_supported") is True
        ):
            reasons.append("unmatched_high_evidence_anchor")
        if not reasons:
            continue
        selected.append(anchor)
        triggers.append(
            {
                "cross_id": cross_id,
                "reasons": sorted(set(reasons)),
                "cross_center_gap_ratio": (
                    round(center_gap, 6) if center_gap is not None else None
                ),
            }
        )
    return {
        "trigger_count": len(triggers),
        "anchors": selected,
        "triggers": triggers,
        "min_center_gap_ratio": min_center_gap_ratio,
        "policy": "Only actionable first-pass anomalies receive a local LLM2 retry.",
    }


def _map_bbox_to_crop(bbox: list[float], crop_bbox: list[float]) -> list[float]:
    crop_width = crop_bbox[2] - crop_bbox[0]
    crop_height = crop_bbox[3] - crop_bbox[1]
    return [
        (bbox[0] - crop_bbox[0]) / crop_width,
        (bbox[1] - crop_bbox[1]) / crop_height,
        (bbox[2] - crop_bbox[0]) / crop_width,
        (bbox[3] - crop_bbox[1]) / crop_height,
    ]


def _map_bbox_from_crop(bbox: list[float], crop_bbox: list[float]) -> list[float]:
    crop_width = crop_bbox[2] - crop_bbox[0]
    crop_height = crop_bbox[3] - crop_bbox[1]
    return [
        crop_bbox[0] + bbox[0] * crop_width,
        crop_bbox[1] + bbox[1] * crop_height,
        crop_bbox[0] + bbox[2] * crop_width,
        crop_bbox[1] + bbox[3] * crop_height,
    ]


def write_cross_anchor_retry_crop(
    image_path: Path,
    output_path: Path,
    anchor: dict,
    *,
    padding_ratio: float,
) -> dict:
    bbox = anchor["bbox"]
    crop_bbox = [
        max(0.0, bbox[0] - padding_ratio),
        max(0.0, bbox[1] - padding_ratio),
        min(1.0, bbox[2] + padding_ratio),
        min(1.0, bbox[3] + padding_ratio),
    ]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    pixel_bbox = (
        round(crop_bbox[0] * image.width),
        round(crop_bbox[1] * image.height),
        round(crop_bbox[2] * image.width),
        round(crop_bbox[3] * image.height),
    )
    crop = image.crop(pixel_bbox)
    mapped_anchor = {
        **anchor,
        "bbox": _map_bbox_to_crop(bbox, crop_bbox),
        "image_scope": "local_retry_crop",
    }
    draw = ImageDraw.Draw(crop)
    local_bbox = mapped_anchor["bbox"]
    draw.rectangle(
        (
            local_bbox[0] * crop.width,
            local_bbox[1] * crop.height,
            local_bbox[2] * crop.width,
            local_bbox[3] * crop.height,
        ),
        outline="blue",
        width=max(2, round(min(crop.width, crop.height) * 0.005)),
    )
    draw.text(
        (local_bbox[0] * crop.width, local_bbox[1] * crop.height),
        f"C{anchor['cross_id']}",
        fill="blue",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, format="JPEG", quality=92)
    return {
        "source_crop_bbox": crop_bbox,
        "anchor": mapped_anchor,
    }


def map_retry_result_to_source(
    result: CrossAnchoredQuestionResult,
    source_crop_bbox: list[float],
) -> CrossAnchoredQuestionResult:
    return CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    **item.model_dump(),
                    "question_bbox": (
                        _map_bbox_from_crop(item.question_bbox, source_crop_bbox)
                        if item.question_bbox is not None
                        else None
                    ),
                }
                for item in result.items
            ]
        }
    )


def audit_duplicate_anchored_questions(
    result: CrossAnchoredQuestionResult,
    *,
    min_iou: float,
) -> dict:
    matched_items = [
        item
        for item in result.items
        if item.matched and item.question_bbox is not None
    ]
    duplicate_candidates = []
    for index, first in enumerate(matched_items):
        for second in matched_items[index + 1 :]:
            overlap = _bbox_iou(first.question_bbox, second.question_bbox)
            if overlap >= min_iou:
                duplicate_candidates.append(
                    {
                        "cross_ids": sorted([first.cross_id, second.cross_id]),
                        "question_bbox_iou": round(overlap, 6),
                    }
                )
    return {
        "min_iou": min_iou,
        "duplicate_candidates": duplicate_candidates,
        "policy": (
            "Duplicate audit records candidates and never rewrites LLM output."
        ),
    }


def cluster_anchored_question_events(
    result: CrossAnchoredQuestionResult,
    *,
    min_iou: float,
) -> dict:
    remaining = {
        item.cross_id: item
        for item in result.items
        if item.matched and item.question_bbox is not None
    }
    clusters = []
    while remaining:
        first_id = min(remaining)
        cluster = [remaining.pop(first_id)]
        index = 0
        while index < len(cluster):
            current = cluster[index]
            connected_ids = [
                cross_id
                for cross_id, candidate in remaining.items()
                if _bbox_iou(current.question_bbox, candidate.question_bbox)
                >= min_iou
            ]
            for cross_id in connected_ids:
                cluster.append(remaining.pop(cross_id))
            index += 1
        clusters.append(sorted(cluster, key=lambda item: item.cross_id))

    events = []
    for event_id, cluster in enumerate(clusters):
        representative = min(
            cluster,
            key=lambda item: (-item.confidence, item.cross_id),
        )
        events.append(
            {
                "event_id": event_id,
                "cross_ids": [item.cross_id for item in cluster],
                "representative_cross_id": representative.cross_id,
                "question_bboxes": [item.question_bbox for item in cluster],
                "confidence": representative.confidence,
            }
        )
    return {
        "min_iou": min_iou,
        "event_count": len(events),
        "unmatched_cross_ids": sorted(
            item.cross_id for item in result.items if not item.matched
        ),
        "events": events,
        "policy": "IoU connected components are audit-only and preserve raw LLM2 items.",
    }


def cluster_anchored_question_runs(
    runs: list[CrossAnchoredQuestionResult],
    *,
    min_iou: float,
) -> dict:
    observations = [
        {
            "observation_id": f"run-{run_index:03d}-cross-{item.cross_id}",
            "run_index": run_index,
            "cross_id": item.cross_id,
            "question_bbox": item.question_bbox,
            "confidence": item.confidence,
        }
        for run_index, result in enumerate(runs, 1)
        for item in result.items
        if item.matched and item.question_bbox is not None
    ]
    first_pass = cluster_anchored_question_events(runs[0], min_iou=min_iou)
    clusters = []
    base_event_by_cross_id = {}
    observation_by_id = {
        observation["observation_id"]: observation for observation in observations
    }
    for event in first_pass["events"]:
        cluster = [
            observation_by_id[f"run-001-cross-{cross_id}"]
            for cross_id in event["cross_ids"]
        ]
        cluster_index = len(clusters)
        clusters.append(cluster)
        for cross_id in event["cross_ids"]:
            base_event_by_cross_id[cross_id] = cluster_index
    for observation in observations:
        if observation["run_index"] == 1:
            continue
        base_index = base_event_by_cross_id.get(observation["cross_id"])
        if base_index is not None and any(
            _bbox_iou(observation["question_bbox"], existing["question_bbox"])
            >= min_iou
            for existing in clusters[base_index]
        ):
            clusters[base_index].append(observation)
        else:
            clusters.append([observation])

    events = []
    for event_id, cluster in enumerate(clusters):
        representative = min(
            cluster,
            key=lambda observation: (
                -observation["confidence"],
                observation["observation_id"],
            ),
        )
        events.append(
            {
                "event_id": event_id,
                "observation_ids": sorted(
                    observation["observation_id"] for observation in cluster
                ),
                "cross_ids": sorted(
                    {observation["cross_id"] for observation in cluster}
                ),
                "representative_observation_id": representative[
                    "observation_id"
                ],
                "question_bboxes": [
                    observation["question_bbox"] for observation in cluster
                ],
                "confidence": representative["confidence"],
            }
        )
    return {
        "run_count": len(runs),
        "min_iou": min_iou,
        "observation_count": len(observations),
        "event_count": len(events),
        "unmatched_observation_ids": sorted(
            f"run-{run_index:03d}-cross-{item.cross_id}"
            for run_index, result in enumerate(runs, 1)
            for item in result.items
            if not item.matched
        ),
        "events": events,
        "policy": (
            "First-pass event boundaries are preserved; retry observations may join "
            "their own overlapping first-pass event but cannot bridge distinct events."
        ),
    }


def compare_llm2_pass_benefit(first_pass: dict, union: dict) -> dict:
    effective_matched_truth_ids = sorted(
        set(first_pass["matched_truth_ids"]) | set(union["matched_truth_ids"])
    )
    recovered_truth_ids = sorted(
        set(union["matched_truth_ids"]) - set(first_pass["matched_truth_ids"])
    )
    remaining_missed_truth_ids = sorted(
        set(first_pass["missed_truth_ids"]) & set(union["missed_truth_ids"])
    )
    false_event_delta = len(union["false_event_ids"]) - len(
        first_pass["false_event_ids"]
    )
    truth_count = len(effective_matched_truth_ids) + len(remaining_missed_truth_ids)
    effective_truth_recall = (
        len(effective_matched_truth_ids) / truth_count if truth_count else None
    )
    return {
        "first_pass_matched_truth_count": first_pass["matched_truth_count"],
        "union_matched_truth_count": len(effective_matched_truth_ids),
        "union_truth_recall": (
            round(effective_truth_recall, 6)
            if effective_truth_recall is not None
            else None
        ),
        "recovered_truth_ids": recovered_truth_ids,
        "recovered_truth_count": len(recovered_truth_ids),
        "remaining_missed_truth_ids": remaining_missed_truth_ids,
        "first_pass_false_event_count": len(first_pass["false_event_ids"]),
        "union_false_event_count": len(union["false_event_ids"]),
        "additional_false_event_count": max(0, false_event_delta),
        "net_false_event_delta": false_event_delta,
        "first_pass_minimum_matched_truth_coverage": first_pass[
            "minimum_matched_truth_coverage"
        ],
        "union_minimum_matched_truth_coverage": union[
            "minimum_matched_truth_coverage"
        ],
        "truth_recall_delta": round(
            effective_truth_recall - first_pass["truth_recall"],
            6,
        ),
    }


def compare_question_events_to_truth(
    event_audit: dict,
    truth_regions: list[dict],
    *,
    min_iou: float,
) -> dict:
    assignments = []
    matched_truth_ids = set()
    false_event_ids = []
    for event in event_audit["events"]:
        scored_truths = [
            (
                *max(
                    (
                        _bbox_iou(bbox, truth["source_bbox_normalized"]),
                        bbox,
                    )
                    for bbox in event["question_bboxes"]
                ),
                truth,
            )
            for truth in truth_regions
        ]
        best_iou, best_bbox, best_truth = max(
            scored_truths,
            key=lambda entry: (entry[0], entry[2]["truth_id"]),
            default=(0.0, None, None),
        )
        truth_id = (
            best_truth["truth_id"]
            if best_truth is not None and best_iou >= min_iou
            else None
        )
        if truth_id is None:
            false_event_ids.append(event["event_id"])
        else:
            matched_truth_ids.add(truth_id)
        truth_coverage = 0.0
        if best_truth is not None and best_bbox is not None:
            truth_area = _bbox_area(best_truth["source_bbox_normalized"])
            if truth_area:
                truth_coverage = _bbox_intersection_area(
                    best_bbox,
                    best_truth["source_bbox_normalized"],
                ) / truth_area
        assignments.append(
            {
                "event_id": event["event_id"],
                "truth_id": truth_id,
                "best_iou": round(best_iou, 6),
                "truth_coverage": round(truth_coverage, 6),
            }
        )
    duplicate_truth_event_ids = []
    for truth_id in matched_truth_ids:
        truth_assignments = [
            assignment
            for assignment in assignments
            if assignment["truth_id"] == truth_id
        ]
        truth_assignments.sort(
            key=lambda assignment: (
                -assignment["best_iou"],
                assignment["event_id"],
            )
        )
        duplicate_truth_event_ids.extend(
            assignment["event_id"] for assignment in truth_assignments[1:]
        )
    false_event_ids = sorted(
        set(false_event_ids) | set(duplicate_truth_event_ids)
    )
    ordered_truth_ids = [truth["truth_id"] for truth in truth_regions]
    matched = [truth_id for truth_id in ordered_truth_ids if truth_id in matched_truth_ids]
    missed = [truth_id for truth_id in ordered_truth_ids if truth_id not in matched_truth_ids]
    truth_count = len(truth_regions)
    matched_coverages = [
        assignment["truth_coverage"]
        for assignment in assignments
        if assignment["truth_id"] is not None
        and assignment["event_id"] not in duplicate_truth_event_ids
    ]
    return {
        "truth_count": truth_count,
        "event_count": len(event_audit["events"]),
        "matched_truth_count": len(matched),
        "truth_recall": round(len(matched) / truth_count, 6) if truth_count else None,
        "min_iou": min_iou,
        "matched_truth_ids": matched,
        "missed_truth_ids": missed,
        "false_event_ids": false_event_ids,
        "duplicate_truth_event_ids": sorted(duplicate_truth_event_ids),
        "minimum_matched_truth_coverage": (
            min(matched_coverages) if matched_coverages else None
        ),
        "assignments": assignments,
    }


def compare_anchored_questions_to_truth(
    result: CrossAnchoredQuestionResult,
    truth_regions: list[dict],
    *,
    min_iou: float,
) -> dict:
    items = []
    matched_truth_ids = set()
    for item in result.items:
        if not item.matched or item.question_bbox is None:
            continue
        scored_truths = [
            (
                _bbox_iou(item.question_bbox, truth["source_bbox_normalized"]),
                truth,
            )
            for truth in truth_regions
        ]
        best_iou, best_truth = max(
            scored_truths,
            key=lambda pair: (pair[0], pair[1]["truth_id"]),
            default=(0.0, None),
        )
        meets_threshold = best_truth is not None and best_iou >= min_iou
        truth_id = best_truth["truth_id"] if meets_threshold else None
        if truth_id is not None:
            matched_truth_ids.add(truth_id)
        items.append(
            {
                "cross_id": item.cross_id,
                "truth_id": truth_id,
                "best_iou": round(best_iou, 6),
                "meets_threshold": meets_threshold,
            }
        )
    ordered_truth_ids = [truth["truth_id"] for truth in truth_regions]
    matched = [truth_id for truth_id in ordered_truth_ids if truth_id in matched_truth_ids]
    missed = [truth_id for truth_id in ordered_truth_ids if truth_id not in matched_truth_ids]
    duplicate_truth_candidates = []
    for truth_id in ordered_truth_ids:
        cross_ids = sorted(
            item["cross_id"]
            for item in items
            if item["truth_id"] == truth_id
        )
        if len(cross_ids) > 1:
            duplicate_truth_candidates.append(
                {"truth_id": truth_id, "cross_ids": cross_ids}
            )
    eligible_pairs = []
    matched_items = [
        item
        for item in result.items
        if item.matched and item.question_bbox is not None
    ]
    for item in matched_items:
        for truth in truth_regions:
            overlap = _bbox_iou(
                item.question_bbox,
                truth["source_bbox_normalized"],
            )
            if overlap >= min_iou:
                eligible_pairs.append(
                    (overlap, item.cross_id, truth["truth_id"])
                )
    assigned_cross_ids = set()
    assigned_truth_ids = set()
    one_to_one_assignments = []
    for overlap, cross_id, truth_id in sorted(
        eligible_pairs,
        key=lambda pair: (-pair[0], pair[1], pair[2]),
    ):
        if cross_id in assigned_cross_ids or truth_id in assigned_truth_ids:
            continue
        assigned_cross_ids.add(cross_id)
        assigned_truth_ids.add(truth_id)
        one_to_one_assignments.append(
            {
                "cross_id": cross_id,
                "truth_id": truth_id,
                "iou": round(overlap, 6),
            }
        )
    one_to_one_assignments.sort(key=lambda item: item["cross_id"])
    unassigned_matched_cross_ids = sorted(
        item.cross_id
        for item in matched_items
        if item.cross_id not in assigned_cross_ids
    )
    truth_count = len(truth_regions)
    return {
        "truth_count": truth_count,
        "matched_truth_count": len(matched),
        "truth_recall": round(len(matched) / truth_count, 6) if truth_count else None,
        "min_iou": min_iou,
        "matched_truth_ids": matched,
        "missed_truth_ids": missed,
        "duplicate_truth_candidates": duplicate_truth_candidates,
        "one_to_one_assignments": one_to_one_assignments,
        "unassigned_matched_cross_ids": unassigned_matched_cross_ids,
        "items": items,
    }


def compare_localized_questions_to_truth(
    localized_questions: list[dict],
    truth_regions: list[dict],
    *,
    min_iou: float,
) -> dict:
    items = []
    matched_truth_ids = set()
    for localized in localized_questions:
        scored_truths = [
            (
                _bbox_iou(localized["bbox"], truth["source_bbox_normalized"]),
                truth,
            )
            for truth in truth_regions
        ]
        best_iou, best_truth = max(
            scored_truths,
            key=lambda pair: (pair[0], pair[1]["truth_id"]),
            default=(0.0, None),
        )
        meets_threshold = best_truth is not None and best_iou >= min_iou
        truth_id = best_truth["truth_id"] if meets_threshold else None
        if truth_id is not None:
            matched_truth_ids.add(truth_id)
        items.append(
            {
                "mark_id": localized["mark_id"],
                "truth_id": truth_id,
                "best_iou": round(best_iou, 6),
                "meets_threshold": meets_threshold,
            }
        )
    ordered_truth_ids = [truth["truth_id"] for truth in truth_regions]
    matched = [truth_id for truth_id in ordered_truth_ids if truth_id in matched_truth_ids]
    missed = [truth_id for truth_id in ordered_truth_ids if truth_id not in matched_truth_ids]
    truth_count = len(truth_regions)
    return {
        "truth_count": truth_count,
        "matched_truth_count": len(matched),
        "truth_recall": round(len(matched) / truth_count, 6) if truth_count else None,
        "min_iou": min_iou,
        "matched_truth_ids": matched,
        "missed_truth_ids": missed,
        "items": items,
    }


def find_duplicate_primitive_candidates(
    primitives: list[ErrorMark],
    *,
    containment_threshold: float,
) -> list[dict]:
    candidates = []
    for index, first in enumerate(primitives):
        for second in primitives[index + 1 :]:
            if first.mark_type != second.mark_type:
                continue
            intersection = _bbox_intersection_area(first.bbox, second.bbox)
            smaller_area = min(_bbox_area(first.bbox), _bbox_area(second.bbox))
            containment = intersection / smaller_area if smaller_area else 0.0
            if containment >= containment_threshold:
                candidates.append(
                    {
                        "primitive_ids": [first.mark_id, second.mark_id],
                        "mark_type": first.mark_type,
                        "intersection_over_smaller_area": round(containment, 6),
                    }
                )
    return candidates


def find_duplicate_event_candidates(
    result: StableEventResult,
    *,
    containment_threshold: float,
) -> list[dict]:
    candidates = []
    for index, first in enumerate(result.events):
        for second in result.events[index + 1 :]:
            intersection = _bbox_intersection_area(first.bbox, second.bbox)
            smaller_area = min(_bbox_area(first.bbox), _bbox_area(second.bbox))
            containment = intersection / smaller_area if smaller_area else 0.0
            if containment >= containment_threshold:
                candidates.append(
                    {
                        "event_ids": [first.event_id, second.event_id],
                        "intersection_over_smaller_area": round(containment, 6),
                    }
                )
    return candidates


def audit_cross_circle_geometry(
    result: StableEventResult,
    *,
    max_center_distance: float,
) -> list[dict]:
    candidates = []
    for event in result.events:
        if event.event_type != "cross_circle":
            continue
        cross_center = (
            (event.cross_bbox[0] + event.cross_bbox[2]) / 2,
            (event.cross_bbox[1] + event.cross_bbox[3]) / 2,
        )
        circle_center = (
            (event.circle_bbox[0] + event.circle_bbox[2]) / 2,
            (event.circle_bbox[1] + event.circle_bbox[3]) / 2,
        )
        center_distance = math.hypot(
            cross_center[0] - circle_center[0],
            cross_center[1] - circle_center[1],
        )
        if center_distance > max_center_distance:
            candidates.append(
                {
                    "event_id": event.event_id,
                    "primitive_ids": list(event.primitive_ids),
                    "center_distance": round(center_distance, 6),
                    "max_center_distance": max_center_distance,
                }
            )
    return candidates


def audit_stable_events_against_cv(
    result: StableEventResult,
    red_mask: np.ndarray,
    components: list[dict],
    *,
    min_red_pixels: int,
    min_red_ratio: float,
) -> dict:
    height, width = red_mask.shape
    event_audits = []
    covered_component_ids: set[int] = set()
    event_count = len(result.events)
    for event in result.events:
        left, top, right, bottom = _pixel_bbox(event.bbox, width, height)
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        crop = red_mask[top:bottom, left:right]
        red_pixel_count = int(crop.sum())
        red_pixel_ratio = red_pixel_count / max(1, crop.size)
        matched_component_ids = sorted(
            component["region_id"]
            for component in components
            if _bbox_intersection_area(event.bbox, component["bbox"]) > 0
        )
        covered_component_ids.update(matched_component_ids)
        if red_pixel_count >= min_red_pixels and red_pixel_ratio >= min_red_ratio:
            status = "supported"
        elif red_pixel_count > 0:
            status = "weak"
        else:
            status = "rejected"
        event_audits.append(
            {
                "event_id": event.event_id,
                "status": status,
                "red_pixel_count": red_pixel_count,
                "red_pixel_ratio": round(red_pixel_ratio, 6),
                "matched_component_ids": matched_component_ids,
            }
        )
    all_component_ids = {component["region_id"] for component in components}
    return {
        "event_count_before": event_count,
        "event_count_after": len(result.events),
        "events": event_audits,
        "covered_component_ids": sorted(covered_component_ids),
        "uncovered_component_ids": sorted(all_component_ids - covered_component_ids),
        "policy": "CV is audit-only and cannot create, remove, merge, or renumber events.",
    }


class ExchangeRecorder:
    """Write raw model exchanges only when the diagnostic CLI enables it."""

    def __init__(self, output_dir: Path):
        self.calls_dir = output_dir / "llm-calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0
        self.event_count = 0
        self.current_call_dir: Path | None = None

    def begin_call(self, operation: str, arguments: dict) -> Path:
        self.call_count += 1
        self.event_count = 0
        call_dir = self.calls_dir / f"call-{self.call_count:03d}-{operation}"
        call_dir.mkdir(parents=True, exist_ok=False)
        self.current_call_dir = call_dir
        _write_json(call_dir / "arguments.json", arguments)
        return call_dir

    def __call__(self, event: dict) -> None:
        if self.current_call_dir is None:
            raise RuntimeError("LLM diagnostic event has no active call")
        self.event_count += 1
        kind = str(event.get("kind", "unknown")).replace("/", "-")
        _write_json(
            self.current_call_dir / f"event-{self.event_count:03d}-{kind}.json",
            event,
        )

    def copy_input(self, call_dir: Path, image_path: str) -> None:
        suffix = Path(image_path).suffix.lower() or ".jpg"
        shutil.copy2(image_path, call_dir / f"input{suffix}")

    def finish_call(
        self,
        call_dir: Path,
        *,
        result: Any | None = None,
        error: BaseException | None = None,
    ) -> None:
        if result is not None:
            _write_json(call_dir / "result.json", result)
        if error is not None:
            _write_json(
                call_dir / "error.json",
                {
                    "type": type(error).__name__,
                    "code": getattr(error, "code", None),
                    "message": str(error),
                    "diagnostic": getattr(error, "diagnostic", None),
                },
            )
        self.current_call_dir = None


class RecordingVisionClient:
    def __init__(self, client: MiniMaxVisionClient, recorder: ExchangeRecorder):
        self.client = client
        self.recorder = recorder

    def _call(self, operation: str, image_path: str, arguments: dict, function):
        call_dir = self.recorder.begin_call(operation, arguments)
        self.recorder.copy_input(call_dir, image_path)
        try:
            result = function()
        except Exception as exc:
            self.recorder.finish_call(call_dir, error=exc)
            raise
        self.recorder.finish_call(call_dir, result=result)
        return result

    def detect_marks(self, image_path, local_red_regions=None, correction=None):
        return self._call(
            "mark_detection",
            image_path,
            {"local_red_regions": local_red_regions or [], "correction": correction},
            lambda: self.client.detect_marks(image_path, local_red_regions, correction),
        )

    def detect_independent_complete_marks(self, image_path: str) -> MarkDetectionResult:
        diagnostic = {
            "operation": "independent_complete_mark_detection",
            "local_red_region_count": 0,
        }
        image_url = prepare_image_data_url(
            image_path,
            self.client.max_edge,
            self.client.jpeg_quality,
            diagnostic,
        )
        return self._call(
            "independent_complete_mark_detection",
            image_path,
            {"local_red_regions": []},
            lambda: self.client._request(
                {
                    "prompt": INDEPENDENT_COMPLETE_MARK_PROMPT,
                    "image_url": image_url,
                },
                MarkDetectionResult,
                diagnostic,
            ),
        )

    def verify_cross_candidates(
        self,
        image_path: str,
        candidates: list[dict],
    ) -> CrossCandidateVerificationResult:
        prompt = CV_CROSS_VERIFICATION_PROMPT.replace(
            "__CANDIDATES__",
            json.dumps(candidates, ensure_ascii=False, indent=2),
        )
        diagnostic = {
            "operation": "cv_cross_candidate_verification",
            "candidate_count": len(candidates),
        }
        image_url = prepare_image_data_url(
            image_path,
            self.client.max_edge,
            self.client.jpeg_quality,
            diagnostic,
        )
        return self._call(
            "cv_cross_candidate_verification",
            image_path,
            {"candidate_ids": [item["candidate_id"] for item in candidates]},
            lambda: self.client._request(
                {"prompt": prompt, "image_url": image_url},
                CrossCandidateVerificationResult,
                diagnostic,
            ),
        )

    def scan_independent_crosses(self, image_path: str) -> IndependentCrossScanResult:
        diagnostic = {
            "operation": "independent_cross_scan",
            "candidate_count": 0,
        }
        image_url = prepare_image_data_url(
            image_path,
            self.client.max_edge,
            self.client.jpeg_quality,
            diagnostic,
        )
        return self._call(
            "independent_cross_scan",
            image_path,
            {},
            lambda: self.client._request(
                {"prompt": INDEPENDENT_CROSS_SCAN_PROMPT, "image_url": image_url},
                IndependentCrossScanResult,
                diagnostic,
            ),
        )

    def verify_fallback_crosses(
        self,
        image_path: str,
        candidates: list[dict],
    ) -> CrossCandidateVerificationResult:
        prompt = FALLBACK_CROSS_VERIFICATION_PROMPT.replace(
            "__CANDIDATES__",
            json.dumps(candidates, ensure_ascii=False, indent=2),
        )
        diagnostic = {
            "operation": "fallback_cross_candidate_verification",
            "candidate_count": len(candidates),
        }
        image_url = prepare_image_data_url(
            image_path,
            self.client.max_edge,
            self.client.jpeg_quality,
            diagnostic,
        )
        return self._call(
            "fallback_cross_candidate_verification",
            image_path,
            {"candidate_ids": [item["candidate_id"] for item in candidates]},
            lambda: self.client._request(
                {"prompt": prompt, "image_url": image_url},
                CrossCandidateVerificationResult,
                diagnostic,
            ),
        )

    def locate_cross_anchored_questions(
        self,
        image_path: str,
        anchors: list[dict],
        subject_hint: str | None,
    ) -> CrossAnchoredQuestionResult:
        prompt = CROSS_ANCHORED_QUESTION_PROMPT.replace(
            "__CROSSES__",
            json.dumps(anchors, ensure_ascii=False, indent=2),
        )
        diagnostic = {
            "operation": "cross_anchored_question_localization",
            "cross_count": len(anchors),
        }
        image_url = prepare_image_data_url(
            image_path,
            self.client.max_edge,
            self.client.jpeg_quality,
            diagnostic,
        )
        return self._call(
            "cross_anchored_question_localization",
            image_path,
            {
                "cross_ids": [anchor["cross_id"] for anchor in anchors],
                "subject_hint": subject_hint,
            },
            lambda: self.client._request(
                {"prompt": prompt, "image_url": image_url},
                CrossAnchoredQuestionResult,
                diagnostic,
            ),
        )
    def detect_marks_in_regions(self, image_path, region_ids, correction=None):
        return self._call(
            "regional_mark_detection",
            image_path,
            {"region_ids": region_ids, "correction": correction},
            lambda: self.client.detect_marks_in_regions(image_path, region_ids, correction),
        )

    def locate_marked_question_context(self, image_path, error_mark, correction=None):
        return self._call(
            "mark_context_localization",
            image_path,
            {"error_mark": error_mark, "correction": correction},
            lambda: self.client.locate_marked_question_context(
                image_path, error_mark, correction
            ),
        )

    def locate_marked_questions(self, image_path, error_marks, correction=None):
        return self._call(
            "mark_localization",
            image_path,
            {"error_marks": error_marks, "correction": correction},
            lambda: self.client.locate_marked_questions(
                image_path, error_marks, correction
            ),
        )

    def recognize_localized_content(self, image_path, mark_ids, subject_hint=None):
        return self._call(
            "content_recognition",
            image_path,
            {"mark_ids": mark_ids, "subject_hint": subject_hint},
            lambda: self.client.recognize_localized_content(
                image_path, mark_ids, subject_hint
            ),
        )

    def consolidate_stable_events(
        self,
        image_path: str,
        primitives: list[ErrorMark],
    ) -> StableEventResult:
        primitive_payload = [
            {
                "primitive_id": mark.mark_id,
                "mark_type": mark.mark_type,
                "bbox": mark.bbox,
                "confidence": mark.confidence,
            }
            for mark in primitives
        ]
        prompt = STABLE_EVENT_CONSOLIDATION_PROMPT.replace(
            "__PRIMITIVES__",
            json.dumps(primitive_payload, ensure_ascii=False, indent=2),
        )
        diagnostic = {
            "operation": "stable_event_consolidation",
            "primitive_count": len(primitives),
        }
        image_url = prepare_image_data_url(
            image_path,
            self.client.max_edge,
            self.client.jpeg_quality,
            diagnostic,
        )
        return self._call(
            "stable_event_consolidation",
            image_path,
            {"primitive_count": len(primitives)},
            lambda: self.client._request(
                {"prompt": prompt, "image_url": image_url},
                StableEventResult,
                diagnostic,
            ),
        )


def run_stable_event_experiment(
    *,
    image_path: Path,
    case_dir: Path,
    client: RecordingVisionClient,
    cv: dict,
    primitive_duplicate_containment_threshold: float,
    cross_circle_max_center_distance: float,
) -> dict:
    experiment_dir = case_dir / "stable-event-experiment"
    experiment_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    primitive_result = client.detect_independent_complete_marks(str(image_path))
    mark_detection_ms = round((time.perf_counter() - started) * 1000, 2)
    primitives = primitive_result.error_marks
    _write_json(experiment_dir / "independent-primitives.json", primitive_result)
    duplicate_primitive_candidates = find_duplicate_primitive_candidates(
        primitives,
        containment_threshold=primitive_duplicate_containment_threshold,
    )
    _write_json(
        experiment_dir / "duplicate-primitive-candidates.json",
        duplicate_primitive_candidates,
    )
    primitive_entries = [mark.model_dump(mode="json") for mark in primitives]
    primitive_overlay = experiment_dir / "numbered-primitives.jpg"
    _draw_boxes(image_path, primitive_overlay, [("P", primitive_entries, "blue")])

    started = time.perf_counter()
    stable_result = client.consolidate_stable_events(
        str(primitive_overlay),
        primitives,
    )
    event_consolidation_ms = round((time.perf_counter() - started) * 1000, 2)
    _write_json(experiment_dir / "stable-events.json", stable_result)
    assignment = validate_stable_event_assignment(
        stable_result,
        primitive_count=len(primitives),
    )
    _write_json(experiment_dir / "assignment-audit.json", assignment)
    primitive_membership_violations = audit_stable_event_primitive_membership(
        stable_result,
        primitives,
    )
    _write_json(
        experiment_dir / "primitive-membership-audit.json",
        primitive_membership_violations,
    )
    duplicate_event_candidates = find_duplicate_event_candidates(
        stable_result,
        containment_threshold=settings.MARK_DEDUP_IOU_THRESHOLD,
    )
    _write_json(
        experiment_dir / "duplicate-event-candidates.json",
        duplicate_event_candidates,
    )
    cross_circle_geometry_candidates = audit_cross_circle_geometry(
        stable_result,
        max_center_distance=cross_circle_max_center_distance,
    )
    _write_json(
        experiment_dir / "cross-circle-geometry-audit.json",
        cross_circle_geometry_candidates,
    )

    cv_audit = audit_stable_events_against_cv(
        stable_result,
        cv["red_mask"],
        cv["components"],
        min_red_pixels=settings.LOCAL_RED_COMPONENT_MIN_PIXELS,
        min_red_ratio=settings.MARK_RED_PIXEL_MIN_RATIO,
    )
    _write_json(experiment_dir / "cv-post-validation.json", cv_audit)
    stable_entries = [
        {"mark_id": event.event_id, "bbox": event.bbox}
        for event in stable_result.events
    ]
    _draw_boxes(
        image_path,
        experiment_dir / "stable-events-overlay.jpg",
        [("E", stable_entries, "blue")],
    )
    result = {
        "primitive_count": len(primitives),
        "event_count": len(stable_result.events),
        **assignment,
        "duplicate_primitive_candidates": duplicate_primitive_candidates,
        "primitive_membership_violations": primitive_membership_violations,
        "duplicate_event_candidates": duplicate_event_candidates,
        "cross_circle_geometry_candidates": cross_circle_geometry_candidates,
        "uncovered_component_ids": cv_audit["uncovered_component_ids"],
        "cv_event_support": cv_audit["events"],
        "mark_detection_ms": mark_detection_ms,
        "event_consolidation_ms": event_consolidation_ms,
    }
    _write_json(experiment_dir / "summary.json", result)
    return result


def build_summary(
    *,
    label: str,
    expected_count: int | None,
    cv: dict,
    pipeline: dict | None,
    pipeline_truth_comparison: dict | None = None,
    stable_experiment: dict | None = None,
    cross_anchor_experiment: dict | None = None,
) -> dict:
    pipeline_ran = pipeline is not None
    stable_experiment_ran = stable_experiment is not None
    cross_anchor_experiment_ran = cross_anchor_experiment is not None
    pipeline = pipeline or {}
    checkpoints = {
        "expected_error_count": expected_count,
        "cv_raw_component_count": cv.get("raw_component_count"),
        "cv_evidence_group_count": cv.get("evidence_group_count"),
        "llm_mark_primitive_count": pipeline.get("mark_primitive_count"),
        "normalized_mark_event_count": pipeline.get("mark_event_count"),
        "localized_mark_count": pipeline.get("localized_mark_count"),
        "content_item_count": pipeline.get("content_item_count"),
        "pipeline_truth_matched_count": (pipeline_truth_comparison or {}).get(
            "matched_truth_count"
        ),
        "pipeline_truth_recall": (pipeline_truth_comparison or {}).get(
            "truth_recall"
        ),
        "stable_primitive_count": (stable_experiment or {}).get("primitive_count"),
        "stable_event_count": (stable_experiment or {}).get("event_count"),
        "stable_duplicate_primitive_count": (
            len(stable_experiment.get("duplicate_primitive_ids", []))
            if stable_experiment_ran
            else None
        ),
        "stable_duplicate_event_candidate_count": (
            len(stable_experiment.get("duplicate_event_candidates", []))
            if stable_experiment_ran
            else None
        ),
        "stable_duplicate_primitive_candidate_count": (
            len(stable_experiment.get("duplicate_primitive_candidates", []))
            if stable_experiment_ran
            else None
        ),
        "stable_cross_circle_geometry_candidate_count": (
            len(stable_experiment.get("cross_circle_geometry_candidates", []))
            if stable_experiment_ran
            else None
        ),
        "stable_primitive_membership_violation_count": (
            len(stable_experiment.get("primitive_membership_violations", []))
            if stable_experiment_ran
            else None
        ),
        "stable_unassigned_primitive_count": (
            len(stable_experiment.get("unassigned_primitive_ids", []))
            if stable_experiment_ran
            else None
        ),
        "stable_uncovered_component_count": (
            len(stable_experiment.get("uncovered_component_ids", []))
            if stable_experiment_ran
            else None
        ),
        "stable_mark_detection_ms": (stable_experiment or {}).get(
            "mark_detection_ms"
        ),
        "stable_event_consolidation_ms": (stable_experiment or {}).get(
            "event_consolidation_ms"
        ),
        "cross_anchor_cv_candidate_count": (cross_anchor_experiment or {}).get(
            "cv_candidate_count"
        ),
        "cross_anchor_confirmed_cross_count": (cross_anchor_experiment or {}).get(
            "llm1_confirmed_cross_count"
        ),
        "cross_anchor_uncertain_retained_count": (
            cross_anchor_experiment or {}
        ).get("llm1_cv_uncertain_retained_count"),
        "cross_anchor_high_score_retained_count": (
            cross_anchor_experiment or {}
        ).get("llm1_cv_high_score_retained_count"),
        "cross_anchor_rejected_retained_count": (
            cross_anchor_experiment or {}
        ).get("llm1_cv_rejected_retained_count"),
        "cross_anchor_fallback_uncertain_retained_count": (
            cross_anchor_experiment or {}
        ).get("llm1_fallback_uncertain_retained_count"),
        "cross_anchor_fallback_generates_anchors": (
            cross_anchor_experiment or {}
        ).get("llm1_fallback_generates_anchors"),
        "cross_anchor_llm1_verification_run_count": (
            cross_anchor_experiment or {}
        ).get("llm1_verification_run_count"),
        "cross_anchor_llm1_unstable_candidate_count": (
            cross_anchor_experiment or {}
        ).get("llm1_unstable_candidate_count"),
        "cross_anchor_fallback_cross_count": (cross_anchor_experiment or {}).get(
            "llm1_fallback_cross_count"
        ),
        "cross_anchor_fallback_verified_count": (
            cross_anchor_experiment or {}
        ).get("llm1_fallback_verified_count"),
        "cross_anchor_independent_scan_count": (
            cross_anchor_experiment or {}
        ).get("llm1_independent_scan_count"),
        "cross_anchor_independent_supported_count": (
            cross_anchor_experiment or {}
        ).get("llm1_independent_supported_count"),
        "cross_anchor_local_geometry_merge_count": (
            cross_anchor_experiment or {}
        ).get("llm1_local_geometry_merge_count"),
        "cross_anchor_llm1_truth_recall": (cross_anchor_experiment or {}).get(
            "llm1_truth_recall"
        ),
        "cross_anchor_llm1_false_cross_count": (cross_anchor_experiment or {}).get(
            "llm1_false_cross_count"
        ),
        "cross_anchor_llm1_duplicate_truth_candidate_count": (
            cross_anchor_experiment or {}
        ).get("llm1_duplicate_truth_candidate_count"),
        "cross_anchor_matched_question_count": (cross_anchor_experiment or {}).get(
            "llm2_matched_question_count"
        ),
        "cross_anchor_llm2_localization_run_count": (
            cross_anchor_experiment or {}
        ).get("llm2_localization_run_count"),
        "cross_anchor_llm2_retry_trigger_count": (
            cross_anchor_experiment or {}
        ).get("llm2_retry_trigger_count"),
        "cross_anchor_llm2_retry_request_count": (
            cross_anchor_experiment or {}
        ).get("llm2_retry_request_count"),
        "cross_anchor_geometry_violation_count": (
            cross_anchor_experiment or {}
        ).get("geometry_violation_count"),
        "cross_anchor_duplicate_question_candidate_count": (
            cross_anchor_experiment or {}
        ).get("duplicate_question_candidate_count"),
        "cross_anchor_duplicate_truth_candidate_count": (
            cross_anchor_experiment or {}
        ).get("duplicate_truth_candidate_count"),
        "cross_anchor_truth_matched_count": (cross_anchor_experiment or {}).get(
            "truth_matched_count"
        ),
        "cross_anchor_truth_recall": (cross_anchor_experiment or {}).get(
            "truth_recall"
        ),
        "cross_anchor_unassigned_matched_cross_count": (
            cross_anchor_experiment or {}
        ).get("unassigned_matched_cross_count"),
        "cross_anchor_stable_question_event_count": (
            cross_anchor_experiment or {}
        ).get("stable_question_event_count"),
        "cross_anchor_first_pass_stable_truth_recall": (
            cross_anchor_experiment or {}
        ).get("first_pass_stable_truth_recall"),
        "cross_anchor_stable_truth_matched_count": (
            cross_anchor_experiment or {}
        ).get("stable_truth_matched_count"),
        "cross_anchor_stable_truth_recall": (
            cross_anchor_experiment or {}
        ).get("stable_truth_recall"),
        "cross_anchor_stable_false_event_count": (
            cross_anchor_experiment or {}
        ).get("stable_false_event_count"),
        "cross_anchor_second_pass_recovered_truth_count": (
            cross_anchor_experiment or {}
        ).get("llm2_second_pass_recovered_truth_count"),
        "cross_anchor_second_pass_recovered_truth_ids": (
            cross_anchor_experiment or {}
        ).get("llm2_second_pass_recovered_truth_ids"),
        "cross_anchor_second_pass_additional_false_event_count": (
            cross_anchor_experiment or {}
        ).get("llm2_second_pass_additional_false_event_count"),
        "cross_anchor_first_pass_minimum_truth_coverage": (
            cross_anchor_experiment or {}
        ).get("first_pass_minimum_matched_truth_coverage"),
        "cross_anchor_stable_minimum_truth_coverage": (
            cross_anchor_experiment or {}
        ).get("stable_minimum_matched_truth_coverage"),
        "cross_anchor_llm_request_count": (cross_anchor_experiment or {}).get(
            "llm_request_count"
        ),
        "cross_anchor_stage_timings_ms": (cross_anchor_experiment or {}).get(
            "timings_ms"
        ),
        "cross_anchor_content_ocr_status": (
            (cross_anchor_experiment or {}).get("content_ocr_status")
            if cross_anchor_experiment_ran
            else None
        ),
    }
    first_divergence = None
    if expected_count is not None and pipeline_ran:
        if checkpoints["normalized_mark_event_count"] != expected_count:
            first_divergence = "mark_detection_or_grouping"
        elif checkpoints["localized_mark_count"] != expected_count:
            first_divergence = "localization"
        elif checkpoints["content_item_count"] != expected_count:
            first_divergence = "content_recognition"
    return {
        "label": label,
        "pipeline_status": "completed_or_failed" if pipeline_ran else "not_run",
        "stable_experiment_status": (
            "completed" if stable_experiment_ran else "not_run"
        ),
        "cross_anchor_experiment_status": (
            "completed" if cross_anchor_experiment_ran else "not_run"
        ),
        "checkpoints": checkpoints,
        "first_count_divergence": first_divergence,
        "note": (
            "CV component/group counts are evidence diagnostics, not error-event counts; "
            "they must not be interpreted as CV success or failure by count alone."
        ),
    }


def _pipeline_arguments() -> dict:
    return {
        "mark_confidence_threshold": settings.MINIMAX_MARK_CONFIDENCE_THRESHOLD,
        "red_pixel_min_ratio": settings.MARK_RED_PIXEL_MIN_RATIO,
        "red_pixel_expansion_ratio": settings.MARK_RED_PIXEL_EXPANSION_RATIO,
        "pair_max_distance_ratio": settings.MARK_PAIR_MAX_DISTANCE_RATIO,
        "pair_max_relative_distance_ratio": settings.MARK_PAIR_MAX_RELATIVE_DISTANCE_RATIO,
        "pair_min_margin_ratio": settings.MARK_PAIR_MIN_MARGIN_RATIO,
        "dedup_iou_threshold": settings.MARK_DEDUP_IOU_THRESHOLD,
        "crop_context_padding_ratio": settings.QUESTION_CROP_CONTEXT_PADDING_RATIO,
        "circle_context_padding_ratio": settings.MARK_CIRCLE_CONTEXT_PADDING_RATIO,
        "evidence_context_min_width_ratio": settings.MARK_EVIDENCE_CONTEXT_MIN_WIDTH_RATIO,
        "evidence_context_min_height_ratio": settings.MARK_EVIDENCE_CONTEXT_MIN_HEIGHT_RATIO,
        "answer_min_circle_overlap_ratio": settings.MARK_ANSWER_MIN_CIRCLE_OVERLAP_RATIO,
        "answer_min_answer_overlap_ratio": settings.MARK_ANSWER_MIN_ANSWER_OVERLAP_RATIO,
        "answer_hard_min_circle_coverage_ratio": settings.MARK_ANSWER_HARD_MIN_CIRCLE_COVERAGE_RATIO,
        "answer_max_center_offset_ratio": settings.MARK_ANSWER_MAX_CENTER_OFFSET_RATIO,
        "answer_max_overflow_ratio": settings.MARK_ANSWER_MAX_OVERFLOW_RATIO,
        "localization_edge_margin_ratio": settings.MARK_LOCALIZATION_EDGE_MARGIN_RATIO,
        "local_red_group_max_gap_ratio": settings.LOCAL_RED_GROUP_MAX_GAP_RATIO,
        "local_red_group_max_area_ratio": settings.LOCAL_RED_COMPONENT_MAX_AREA_RATIO,
        "image_max_edge": settings.MINIMAX_IMAGE_MAX_EDGE,
        "image_jpeg_quality": settings.MINIMAX_IMAGE_JPEG_QUALITY,
        "image_max_pixels": settings.QUESTION_IMAGE_MAX_PIXELS,
        "mark_stage_retry_count": settings.MINIMAX_MARK_STAGE_RETRY_COUNT,
        "localization_stage_retry_count": settings.MINIMAX_LOCALIZATION_STAGE_RETRY_COUNT,
        "content_stage_retry_count": settings.MINIMAX_CONTENT_STAGE_RETRY_COUNT,
        "content_batch_size": settings.MINIMAX_CONTENT_BATCH_SIZE,
        "local_red_rescue_min_pixels": settings.LOCAL_RED_RESCUE_MIN_PIXELS,
    }


def run_case(
    *,
    label: str,
    image_path: Path,
    output_dir: Path,
    expected_count: int | None,
    subject_hint: str | None,
    cv_only: bool,
    cross_cv_config: dict | None = None,
    truth_regions: list[dict] | None = None,
    compare_stable_events: bool = False,
    compare_cross_anchor: bool = False,
    primitive_duplicate_containment_threshold: float | None = None,
    cross_circle_max_center_distance: float | None = None,
) -> dict:
    case_started = time.perf_counter()
    timings_ms = {}
    case_dir = output_dir / label
    case_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(image_path, case_dir / ("source" + image_path.suffix.lower()))
    phase_started = time.perf_counter()
    cv = write_cv_artifacts(
        image_path,
        case_dir,
        max_edge=settings.LOCAL_RED_SCAN_MAX_EDGE,
        min_component_pixels=settings.LOCAL_RED_COMPONENT_MIN_PIXELS,
        max_component_area_ratio=settings.LOCAL_RED_COMPONENT_MAX_AREA_RATIO,
        max_thinness_ratio=settings.LOCAL_RED_COMPONENT_MAX_THINNESS_RATIO,
        group_max_gap_ratio=settings.LOCAL_RED_GROUP_MAX_GAP_RATIO,
        group_max_area_ratio=settings.LOCAL_RED_COMPONENT_MAX_AREA_RATIO,
    )
    timings_ms["red_evidence_cv"] = round(
        (time.perf_counter() - phase_started) * 1000, 2
    )
    phase_started = time.perf_counter()
    cross_cv_experiment = (
        write_cross_cv_artifacts(
            image_path,
            case_dir,
            cross_cv_config,
            truth_regions,
        )
        if cross_cv_config is not None
        else None
    )
    timings_ms["cross_candidate_cv"] = round(
        (time.perf_counter() - phase_started) * 1000, 2
    )
    pipeline_diagnostic = None
    pipeline_truth_comparison = None
    stable_experiment = None
    stable_experiment_error = None
    cross_anchor_experiment = None
    cross_anchor_experiment_error = None
    error = None
    if not cv_only:
        recorder = ExchangeRecorder(case_dir)
        client = MiniMaxVisionClient.from_settings()
        client.diagnostic_event_sink = recorder
        recording_client = RecordingVisionClient(client, recorder)
        arguments = _pipeline_arguments()
        _write_json(case_dir / "effective-config.json", arguments)
        phase_started = time.perf_counter()
        try:
            result, localizations, marks, pipeline_diagnostic = recognize_marked_three_stage(
                client=recording_client,
                image_path=str(image_path),
                subject_hint=subject_hint,
                local_red_regions=[list(region.bbox) for region in cv["scan_result"].regions],
                local_red_evidence_regions=list(cv["scan_result"].regions),
                **arguments,
            )
            _write_json(case_dir / "pipeline-result.json", result)
            _write_json(case_dir / "pipeline-localizations.json", localizations)
            _write_json(case_dir / "pipeline-marks.json", marks)
            _write_json(case_dir / "pipeline-diagnostic.json", pipeline_diagnostic)
            mark_entries = [mark.model_dump(mode="json") for mark in marks]
            location_entries = [
                {"mark_id": mark_id, "bbox": item.bbox}
                for mark_id, item in sorted(localizations.items())
            ]
            _draw_boxes(
                image_path,
                case_dir / "pipeline-overlay.jpg",
                [("M", mark_entries, "red"), ("Q", location_entries, "blue")],
            )
            if cross_cv_config is not None and truth_regions is not None:
                pipeline_truth_comparison = compare_localized_questions_to_truth(
                    location_entries,
                    truth_regions,
                    min_iou=float(cross_cv_config["question_truth_min_iou"]),
                )
                _write_json(
                    case_dir / "pipeline-truth-comparison.json",
                    pipeline_truth_comparison,
                )
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "message": str(exc),
                "diagnostic": getattr(exc, "diagnostic", None),
            }
            pipeline_diagnostic = getattr(exc, "diagnostic", None)
            _write_json(case_dir / "pipeline-error.json", error)
        finally:
            timings_ms["production_pipeline"] = round(
                (time.perf_counter() - phase_started) * 1000, 2
            )
        if compare_stable_events:
            phase_started = time.perf_counter()
            try:
                if primitive_duplicate_containment_threshold is None:
                    raise ValueError(
                        "primitive duplicate containment threshold is required"
                    )
                if cross_circle_max_center_distance is None:
                    raise ValueError("cross-circle max center distance is required")
                stable_experiment = run_stable_event_experiment(
                    image_path=image_path,
                    case_dir=case_dir,
                    client=recording_client,
                    cv=cv,
                    primitive_duplicate_containment_threshold=(
                        primitive_duplicate_containment_threshold
                    ),
                    cross_circle_max_center_distance=(
                        cross_circle_max_center_distance
                    ),
                )
            except Exception as exc:
                stable_experiment_error = {
                    "type": type(exc).__name__,
                    "code": getattr(exc, "code", None),
                    "message": str(exc),
                    "diagnostic": getattr(exc, "diagnostic", None),
                }
                _write_json(
                    case_dir / "stable-event-experiment-error.json",
                    stable_experiment_error,
                )
            finally:
                timings_ms["stable_event_experiment"] = round(
                    (time.perf_counter() - phase_started) * 1000, 2
                )
        if compare_cross_anchor:
            phase_started = time.perf_counter()
            try:
                if cross_cv_config is None or truth_regions is None:
                    raise ValueError(
                        "cross-anchor comparison requires CV config and truth regions"
                    )
                cv_experiment_dir = case_dir / "cv-cross-experiment"
                cv_candidates = json.loads(
                    (cv_experiment_dir / "candidates.json").read_text(encoding="utf-8")
                )
                cross_anchor_experiment = run_cross_anchor_experiment(
                    image_path=image_path,
                    case_dir=case_dir,
                    client=recording_client,
                    cv_candidates=cv_candidates,
                    candidate_overlay_path=(
                        cv_experiment_dir / "candidates-overlay.jpg"
                    ),
                    truth_regions=truth_regions,
                    config=cross_cv_config,
                    subject_hint=subject_hint,
                )
            except Exception as exc:
                cross_anchor_experiment_error = {
                    "type": type(exc).__name__,
                    "code": getattr(exc, "code", None),
                    "message": str(exc),
                    "diagnostic": getattr(exc, "diagnostic", None),
                }
                _write_json(
                    case_dir / "cross-anchor-experiment-error.json",
                    cross_anchor_experiment_error,
                )
            finally:
                timings_ms["cross_anchor_experiment"] = round(
                    (time.perf_counter() - phase_started) * 1000, 2
                )
    timings_ms["total"] = round((time.perf_counter() - case_started) * 1000, 2)
    summary = build_summary(
        label=label,
        expected_count=expected_count,
        cv=cv,
        pipeline=pipeline_diagnostic,
        pipeline_truth_comparison=pipeline_truth_comparison,
        stable_experiment=stable_experiment,
        cross_anchor_experiment=cross_anchor_experiment,
    )
    summary["error"] = error
    summary["stable_experiment_error"] = stable_experiment_error
    summary["cross_anchor_experiment_error"] = cross_anchor_experiment_error
    summary["cv_cross_experiment"] = cross_cv_experiment
    summary["timings_ms"] = timings_ms
    _write_json(case_dir / "timings.json", timings_ms)
    _write_json(case_dir / "summary.json", summary)
    return summary


def run_cross_anchor_experiment(
    *,
    image_path: Path,
    case_dir: Path,
    client,
    cv_candidates: list[dict],
    candidate_overlay_path: Path,
    truth_regions: list[dict],
    config: dict,
    subject_hint: str | None,
) -> dict:
    experiment_started = time.perf_counter()
    timings_ms = {}
    experiment_dir = case_dir / "cross-anchor-experiment"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(candidate_overlay_path, experiment_dir / "cv-candidates-overlay.jpg")
    candidate_montage_path = experiment_dir / "candidate-montage.jpg"
    phase_started = time.perf_counter()
    montage_summary = write_cross_candidate_montage(
        image_path,
        candidate_montage_path,
        cv_candidates,
        full_page_max_edge=int(config["cross_anchor_montage_full_page_max_edge"]),
        tile_edge=int(config["cross_anchor_montage_tile_edge"]),
        columns=int(config["cross_anchor_montage_columns"]),
        crop_padding_ratio=float(
            config["cross_anchor_montage_crop_padding_ratio"]
        ),
    )
    _write_json(experiment_dir / "candidate-montage.json", montage_summary)
    timings_ms["candidate_montage"] = round(
        (time.perf_counter() - phase_started) * 1000, 2
    )

    candidate_ids = [candidate["candidate_id"] for candidate in cv_candidates]
    verification_runs = []
    phase_started = time.perf_counter()
    verification_run_count = int(
        config.get("cross_anchor_llm1_verification_runs", 1)
    )
    for run_index in range(1, verification_run_count + 1):
        run_result = client.verify_cross_candidates(
            str(candidate_montage_path),
            cv_candidates,
        )
        _write_json(
            experiment_dir
            / f"llm1-candidate-verification-run-{run_index:03d}.json",
            run_result,
        )
        run_audit = audit_cross_candidate_dispositions(run_result, candidate_ids)
        if not run_audit["valid"]:
            _write_json(
                experiment_dir
                / f"llm1-candidate-membership-audit-run-{run_index:03d}.json",
                run_audit,
            )
            _write_json(
                experiment_dir / "llm1-candidate-membership-audit.json",
                run_audit,
            )
            raise ValueError("candidate disposition audit failed")
        verification_runs.append(run_result)
    timings_ms["llm1_candidate_verification"] = round(
        (time.perf_counter() - phase_started) * 1000, 2
    )
    verification, stability_audit = aggregate_cross_candidate_verifications(
        verification_runs
    )
    _write_json(experiment_dir / "llm1-candidate-verification.json", verification)
    _write_json(
        experiment_dir / "llm1-candidate-stability-audit.json",
        stability_audit,
    )
    candidate_audit = audit_cross_candidate_dispositions(
        verification,
        candidate_ids,
    )
    _write_json(
        experiment_dir / "llm1-candidate-membership-audit.json",
        candidate_audit,
    )
    if not candidate_audit["valid"]:
        raise ValueError("candidate disposition audit failed")

    phase_started = time.perf_counter()
    independent_scan = client.scan_independent_crosses(str(image_path))
    _write_json(experiment_dir / "llm1-independent-scan.json", independent_scan)
    timings_ms["independent_cross_scan"] = round(
        (time.perf_counter() - phase_started) * 1000, 2
    )
    fallback_candidates = [
        {
            "candidate_id": candidate_id,
            "bbox": list(cross.bbox),
            "center": [
                round((cross.bbox[0] + cross.bbox[2]) / 2, 6),
                round((cross.bbox[1] + cross.bbox[3]) / 2, 6),
            ],
        }
        for candidate_id, cross in enumerate(independent_scan.crosses)
    ]
    phase_started = time.perf_counter()
    if fallback_candidates:
        fallback_montage_path = experiment_dir / "fallback-candidate-montage.jpg"
        fallback_montage_summary = write_cross_candidate_montage(
            image_path,
            fallback_montage_path,
            fallback_candidates,
            full_page_max_edge=int(
                config["cross_anchor_montage_full_page_max_edge"]
            ),
            tile_edge=int(config["cross_anchor_montage_tile_edge"]),
            columns=int(config["cross_anchor_montage_columns"]),
            crop_padding_ratio=float(
                config["cross_anchor_montage_crop_padding_ratio"]
            ),
        )
        _write_json(
            experiment_dir / "fallback-candidate-montage.json",
            fallback_montage_summary,
        )
        fallback_verification = client.verify_fallback_crosses(
            str(fallback_montage_path),
            fallback_candidates,
        )
    else:
        fallback_verification = CrossCandidateVerificationResult(verdicts=[])
    timings_ms["fallback_montage_and_verification"] = round(
        (time.perf_counter() - phase_started) * 1000, 2
    )
    _write_json(
        experiment_dir / "llm1-fallback-candidate-verification.json",
        fallback_verification,
    )
    fallback_audit = audit_cross_candidate_dispositions(
        fallback_verification,
        [candidate["candidate_id"] for candidate in fallback_candidates],
    )
    _write_json(
        experiment_dir / "llm1-fallback-candidate-membership-audit.json",
        fallback_audit,
    )
    if not fallback_audit["valid"]:
        raise ValueError("fallback candidate disposition audit failed")
    confirmed_fallback_ids = {
        verdict.candidate_id
        for verdict in fallback_verification.verdicts
        if verdict.disposition == "confirmed"
    }
    uncertain_fallback_ids = {
        verdict.candidate_id
        for verdict in fallback_verification.verdicts
        if verdict.disposition == "uncertain"
        and config.get(
            "cross_anchor_retain_uncertain_fallback_candidates",
            False,
        )
    }
    retained_fallback_ids = confirmed_fallback_ids | uncertain_fallback_ids
    phase_started = time.perf_counter()
    fallback_generates_anchors = bool(
        config.get("cross_anchor_fallback_generates_anchors", True)
    )
    verified_independent_scan = IndependentCrossScanResult(
        crosses=[
            cross
            for candidate_id, cross in enumerate(independent_scan.crosses)
            if fallback_generates_anchors
            and candidate_id in retained_fallback_ids
        ]
    )
    anchors = build_cross_anchors(
        verification,
        verified_independent_scan,
        cv_candidates,
        config,
    )
    anchor_merge_audit = {
        "cv_dedupe_iou_threshold": config[
            "cross_anchor_cv_dedupe_iou_threshold"
        ],
        "cv_dedupe_center_distance_ratio": config[
            "cross_anchor_cv_dedupe_center_distance_ratio"
        ],
        "merged_anchors": [
            {
                "source_candidate_ids": anchor["source_candidate_ids"],
                "bbox": anchor["bbox"],
                "reason": anchor["merge_reason"],
            }
            for anchor in anchors
            if anchor["merge_reason"] is not None
        ],
        "policy": "Only local geometry may merge independently verified CV candidates.",
    }
    _write_json(
        experiment_dir / "llm1-anchor-merge-audit.json",
        anchor_merge_audit,
    )
    _write_json(experiment_dir / "confirmed-crosses.json", anchors)
    llm1_truth_comparison = compare_cross_candidates_to_truth(
        [
            {
                "candidate_id": anchor["cross_id"],
                "bbox": anchor["bbox"],
                "center": [
                    round((anchor["bbox"][0] + anchor["bbox"][2]) / 2, 6),
                    round((anchor["bbox"][1] + anchor["bbox"][3]) / 2, 6),
                ],
            }
            for anchor in anchors
        ],
        truth_regions,
        margin_ratio=float(config.get("truth_match_margin_ratio", 0.0)),
    )
    _write_json(
        experiment_dir / "llm1-truth-comparison.json",
        llm1_truth_comparison,
    )
    llm1_truth_multiplicity = []
    for truth in truth_regions:
        candidate_ids = sorted(
            assignment["candidate_id"]
            for assignment in llm1_truth_comparison["assignments"]
            if assignment["truth_id"] == truth["truth_id"]
        )
        if len(candidate_ids) > 1:
            llm1_truth_multiplicity.append(
                {
                    "truth_id": truth["truth_id"],
                    "cross_ids": candidate_ids,
                }
            )
    _write_json(
        experiment_dir / "llm1-truth-multiplicity-audit.json",
        {
            "duplicate_truth_candidates": llm1_truth_multiplicity,
            "policy": "Truth identities are audit-only and never merge anchors.",
        },
    )
    anchor_entries = [
        {"mark_id": anchor["cross_id"], "bbox": anchor["bbox"]}
        for anchor in anchors
    ]
    anchor_overlay_path = experiment_dir / "confirmed-crosses-overlay.jpg"
    _draw_boxes(
        image_path,
        anchor_overlay_path,
        [("C", anchor_entries, "blue")],
    )
    timings_ms["anchor_build_and_audit"] = round(
        (time.perf_counter() - phase_started) * 1000, 2
    )

    batch_size = int(config["cross_anchor_llm2_batch_size"])
    llm2_run_count = int(config.get("cross_anchor_llm2_localization_runs", 1))
    llm2_batch_count = 0
    llm2_started = time.perf_counter()
    llm2_runs = []
    assignment_audits = []
    geometry_audits = []
    run_started = time.perf_counter()
    anchored_items = []
    for batch_index, start in enumerate(range(0, len(anchors), batch_size), 1):
        llm2_batch_count += 1
        batch = anchors[start : start + batch_size]
        batch_entries = [
            {"mark_id": anchor["cross_id"], "bbox": anchor["bbox"]}
            for anchor in batch
        ]
        batch_overlay_path = experiment_dir / (
            f"llm2-run-001-batch-{batch_index:03d}-overlay.jpg"
        )
        _draw_boxes(
            image_path,
            batch_overlay_path,
            [("C", batch_entries, "blue")],
        )
        batch_result = client.locate_cross_anchored_questions(
            str(batch_overlay_path),
            batch,
            subject_hint,
        )
        anchored_items.extend(batch_result.items)
    first_run = CrossAnchoredQuestionResult(items=anchored_items)
    llm2_runs.append(first_run)
    _write_json(
        experiment_dir / "llm2-anchored-questions-run-001.json",
        first_run,
    )
    first_assignment_audit = audit_cross_anchor_assignments(
        first_run,
        [anchor["cross_id"] for anchor in anchors],
    )
    assignment_audits.append(first_assignment_audit)
    _write_json(
        experiment_dir / "llm2-cross-assignment-audit-run-001.json",
        first_assignment_audit,
    )
    if not first_assignment_audit["valid"]:
        raise ValueError("cross anchor assignment audit failed")
    first_geometry_audit = audit_anchored_question_geometry(
        first_run,
        anchors,
        max_area_ratio=float(config["cross_anchor_question_max_area_ratio"]),
        max_gap_ratio=float(config["cross_anchor_question_max_gap_ratio"]),
    )
    geometry_audits.append(first_geometry_audit)
    _write_json(
        experiment_dir / "question-geometry-audit-run-001.json",
        first_geometry_audit,
    )
    timings_ms["llm2_localization_run_001"] = round(
        (time.perf_counter() - run_started) * 1000,
        2,
    )

    retry_selection = select_llm2_retry_anchors(
        first_run,
        anchors,
        first_geometry_audit,
        min_center_gap_ratio=float(
            config["cross_anchor_llm2_retry_min_center_gap_ratio"]
        ),
    )
    _write_json(experiment_dir / "llm2-retry-selection.json", retry_selection)
    retry_request_count = 0
    if llm2_run_count >= 2:
        retry_started = time.perf_counter()
        retry_items = []
        for anchor in retry_selection["anchors"]:
            retry_request_count += 1
            llm2_batch_count += 1
            cross_id = anchor["cross_id"]
            retry_image_path = experiment_dir / f"llm2-retry-cross-{cross_id:03d}.jpg"
            crop = write_cross_anchor_retry_crop(
                image_path,
                retry_image_path,
                anchor,
                padding_ratio=float(
                    config["cross_anchor_llm2_retry_crop_padding_ratio"]
                ),
            )
            _write_json(
                experiment_dir / f"llm2-retry-cross-{cross_id:03d}-crop.json",
                crop,
            )
            local_result = client.locate_cross_anchored_questions(
                str(retry_image_path),
                [crop["anchor"]],
                subject_hint,
            )
            _write_json(
                experiment_dir / f"llm2-retry-cross-{cross_id:03d}-local.json",
                local_result,
            )
            mapped_result = map_retry_result_to_source(
                local_result,
                crop["source_crop_bbox"],
            )
            retry_items.extend(mapped_result.items)
        retry_run = CrossAnchoredQuestionResult(items=retry_items)
        llm2_runs.append(retry_run)
        _write_json(
            experiment_dir / "llm2-anchored-questions-run-002.json",
            retry_run,
        )
        retry_anchor_ids = [
            anchor["cross_id"] for anchor in retry_selection["anchors"]
        ]
        retry_assignment_audit = audit_cross_anchor_assignments(
            retry_run,
            retry_anchor_ids,
        )
        assignment_audits.append(retry_assignment_audit)
        _write_json(
            experiment_dir / "llm2-cross-assignment-audit-run-002.json",
            retry_assignment_audit,
        )
        if not retry_assignment_audit["valid"]:
            raise ValueError("cross anchor retry assignment audit failed")
        retry_geometry_audit = audit_anchored_question_geometry(
            retry_run,
            retry_selection["anchors"],
            max_area_ratio=float(config["cross_anchor_question_max_area_ratio"]),
            max_gap_ratio=float(config["cross_anchor_question_max_gap_ratio"]),
        )
        geometry_audits.append(retry_geometry_audit)
        _write_json(
            experiment_dir / "question-geometry-audit-run-002.json",
            retry_geometry_audit,
        )
        timings_ms["llm2_localization_run_002"] = round(
            (time.perf_counter() - retry_started) * 1000,
            2,
        )
    timings_ms["llm2_localization"] = round(
        (time.perf_counter() - llm2_started) * 1000, 2
    )
    phase_started = time.perf_counter()
    anchored_questions = llm2_runs[0]
    _write_json(
        experiment_dir / "llm2-anchored-questions.json",
        anchored_questions,
    )
    assignment_audit = assignment_audits[0]
    _write_json(
        experiment_dir / "llm2-cross-assignment-audit.json",
        assignment_audit,
    )
    geometry_audit = {
        "run_count": len(llm2_runs),
        "violations": [
            {"run_index": run_index, **violation}
            for run_index, audit in enumerate(geometry_audits, 1)
            for violation in audit["violations"]
        ],
        "policy": "Geometry violations from every LLM2 run are audit-only.",
    }
    _write_json(
        experiment_dir / "question-geometry-audit.json",
        geometry_audit,
    )
    duplicate_question_audit = audit_duplicate_anchored_questions(
        anchored_questions,
        min_iou=float(config["cross_anchor_duplicate_question_iou_threshold"]),
    )
    _write_json(
        experiment_dir / "duplicate-question-audit.json",
        duplicate_question_audit,
    )
    truth_comparison = compare_anchored_questions_to_truth(
        anchored_questions,
        truth_regions,
        min_iou=float(config["question_truth_min_iou"]),
    )
    _write_json(experiment_dir / "truth-comparison.json", truth_comparison)
    first_pass_stable_events = cluster_anchored_question_events(
        anchored_questions,
        min_iou=float(config["cross_anchor_duplicate_question_iou_threshold"]),
    )
    _write_json(
        experiment_dir / "stable-question-events-first-pass.json",
        first_pass_stable_events,
    )
    first_pass_stable_truth_comparison = compare_question_events_to_truth(
        first_pass_stable_events,
        truth_regions,
        min_iou=float(config["question_truth_min_iou"]),
    )
    _write_json(
        experiment_dir
        / "stable-question-events-first-pass-truth-comparison.json",
        first_pass_stable_truth_comparison,
    )
    stable_question_events = cluster_anchored_question_runs(
        llm2_runs,
        min_iou=float(config["cross_anchor_duplicate_question_iou_threshold"]),
    )
    _write_json(
        experiment_dir / "stable-question-events.json",
        stable_question_events,
    )
    stable_truth_comparison = compare_question_events_to_truth(
        stable_question_events,
        truth_regions,
        min_iou=float(config["question_truth_min_iou"]),
    )
    _write_json(
        experiment_dir / "stable-question-events-truth-comparison.json",
        stable_truth_comparison,
    )
    llm2_pass_benefit = compare_llm2_pass_benefit(
        first_pass_stable_truth_comparison,
        stable_truth_comparison,
    )
    llm2_pass_benefit["second_pass_elapsed_ms"] = timings_ms.get(
        "llm2_localization_run_002"
    )
    llm2_pass_benefit["retry_trigger_count"] = retry_selection["trigger_count"]
    llm2_pass_benefit["retry_request_count"] = retry_request_count
    llm2_pass_benefit["total_llm2_elapsed_ms"] = timings_ms[
        "llm2_localization"
    ]
    _write_json(
        experiment_dir / "llm2-pass-benefit.json",
        llm2_pass_benefit,
    )
    timings_ms["post_llm2_audit"] = round(
        (time.perf_counter() - phase_started) * 1000, 2
    )
    timings_ms["total"] = round(
        (time.perf_counter() - experiment_started) * 1000, 2
    )
    _write_json(experiment_dir / "timings.json", timings_ms)

    matched_questions = [item for item in anchored_questions.items if item.matched]
    disposition_counts = {
        disposition: sum(
            verdict.disposition == disposition for verdict in verification.verdicts
        )
        for disposition in ("confirmed", "rejected", "uncertain")
    }

    summary = {
        "cv_candidate_count": len(cv_candidates),
        "llm1_verification_run_count": verification_run_count,
        "llm1_unstable_candidate_count": len(
            stability_audit["unstable_candidate_ids"]
        ),
        "llm1_confirmed_cross_count": len(anchors),
        "llm1_cv_confirmed_cross_count": sum(
            anchor["source"] == "cv_confirmed" for anchor in anchors
        ),
        "llm1_cv_uncertain_retained_count": sum(
            anchor["source"] == "cv_uncertain" for anchor in anchors
        ),
        "llm1_cv_high_score_retained_count": sum(
            anchor["source"] == "cv_high_score_retained" for anchor in anchors
        ),
        "llm1_cv_rejected_retained_count": sum(
            anchor["source"] == "cv_rejected_retained" for anchor in anchors
        ),
        "llm1_fallback_cross_count": sum(
            anchor["source"] == "llm_fallback" for anchor in anchors
        ),
        "llm1_fallback_verified_count": len(confirmed_fallback_ids),
        "llm1_fallback_uncertain_retained_count": len(uncertain_fallback_ids),
        "llm1_fallback_generates_anchors": fallback_generates_anchors,
        "llm1_independent_scan_count": len(independent_scan.crosses),
        "llm1_independent_supported_count": sum(
            anchor["independent_scan_supported"] for anchor in anchors
        ),
        "llm1_rejected_candidate_count": disposition_counts["rejected"],
        "llm1_uncertain_candidate_count": disposition_counts["uncertain"],
        "llm1_candidate_audit_valid": candidate_audit["valid"],
        "llm1_local_geometry_merge_count": len(
            anchor_merge_audit["merged_anchors"]
        ),
        "llm1_truth_matched_count": llm1_truth_comparison["matched_truth_count"],
        "llm1_truth_recall": llm1_truth_comparison["truth_recall"],
        "llm1_false_cross_count": len(
            llm1_truth_comparison["false_candidate_ids"]
        ),
        "llm1_duplicate_truth_candidate_count": len(
            llm1_truth_multiplicity
        ),
        "llm2_matched_question_count": len(matched_questions),
        "llm2_unmatched_cross_count": len(anchored_questions.items)
        - len(matched_questions),
        "llm2_assignment_audit_valid": assignment_audit["valid"],
        "llm2_localization_run_count": len(llm2_runs),
        "llm2_retry_trigger_count": retry_selection["trigger_count"],
        "llm2_retry_request_count": retry_request_count,
        "geometry_violation_count": len(geometry_audit["violations"]),
        "duplicate_question_candidate_count": len(
            duplicate_question_audit["duplicate_candidates"]
        ),
        "duplicate_truth_candidate_count": len(
            truth_comparison["duplicate_truth_candidates"]
        ),
        "truth_matched_count": truth_comparison["matched_truth_count"],
        "truth_count": truth_comparison["truth_count"],
        "truth_recall": truth_comparison["truth_recall"],
        "unassigned_matched_cross_count": len(
            truth_comparison["unassigned_matched_cross_ids"]
        ),
        "first_pass_stable_question_event_count": first_pass_stable_events[
            "event_count"
        ],
        "first_pass_stable_truth_matched_count": (
            first_pass_stable_truth_comparison["matched_truth_count"]
        ),
        "first_pass_stable_truth_recall": first_pass_stable_truth_comparison[
            "truth_recall"
        ],
        "first_pass_stable_false_event_count": len(
            first_pass_stable_truth_comparison["false_event_ids"]
        ),
        "stable_question_event_count": stable_question_events["event_count"],
        "stable_truth_matched_count": llm2_pass_benefit[
            "union_matched_truth_count"
        ],
        "stable_truth_recall": llm2_pass_benefit["union_truth_recall"],
        "stable_false_event_count": len(
            stable_truth_comparison["false_event_ids"]
        ),
        "llm2_second_pass_recovered_truth_count": llm2_pass_benefit[
            "recovered_truth_count"
        ],
        "llm2_second_pass_recovered_truth_ids": llm2_pass_benefit[
            "recovered_truth_ids"
        ],
        "llm2_second_pass_additional_false_event_count": llm2_pass_benefit[
            "additional_false_event_count"
        ],
        "first_pass_minimum_matched_truth_coverage": llm2_pass_benefit[
            "first_pass_minimum_matched_truth_coverage"
        ],
        "stable_minimum_matched_truth_coverage": llm2_pass_benefit[
            "union_minimum_matched_truth_coverage"
        ],
        "llm_request_count": (
            verification_run_count
            + 1
            + (1 if fallback_candidates else 0)
            + llm2_batch_count
        ),
        "timings_ms": timings_ms,
        "content_ocr_status": "not_run",
    }
    _write_json(experiment_dir / "summary.json", summary)
    return summary


def _parse_labeled_paths(values: list[str]) -> list[tuple[str, Path]]:
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


def _parse_expected(values: list[str]) -> dict[str, int]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected count must use label=count: {value}")
        label, count = value.split("=", 1)
        parsed[label] = int(count)
    return parsed


def load_cross_cv_inputs(
    config_path: Path,
    truth_path: Path,
    labels: list[str],
) -> tuple[dict, dict[str, list[dict]]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "analysis_max_edge",
        "red_min_channel",
        "red_min_excess",
        "axis_line_min_density",
        "arm_inner_radius_ratio",
        "arm_outer_radius_ratio",
        "diagonal_band_ratio",
        "arm_min_density",
        "center_radius_ratio",
        "center_min_density",
        "candidate_merge_radius_ratio",
        "bbox_padding_ratio",
        "truth_match_margin_ratio",
        "cross_anchor_question_max_area_ratio",
        "cross_anchor_question_max_gap_ratio",
        "cross_anchor_duplicate_question_iou_threshold",
        "cross_anchor_llm1_verification_runs",
        "cross_anchor_llm2_localization_runs",
        "cross_anchor_llm2_retry_min_center_gap_ratio",
        "cross_anchor_llm2_retry_crop_padding_ratio",
        "cross_anchor_fallback_generates_anchors",
        "cross_anchor_retain_uncertain_candidates",
        "cross_anchor_retain_rejected_candidates",
        "cross_anchor_retain_uncertain_fallback_candidates",
        "cross_anchor_high_cv_min_arm_density",
        "cross_anchor_high_cv_min_center_density",
        "cross_anchor_cv_dedupe_iou_threshold",
        "cross_anchor_cv_dedupe_center_distance_ratio",
        "cross_anchor_fallback_merge_iou_threshold",
        "cross_anchor_fallback_merge_center_distance_ratio",
        "cross_anchor_montage_full_page_max_edge",
        "cross_anchor_montage_tile_edge",
        "cross_anchor_montage_columns",
        "cross_anchor_montage_crop_padding_ratio",
        "cross_anchor_llm2_batch_size",
        "question_truth_min_iou",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"cross CV config missing fields: {missing}")
    positive_integer_fields = {
        "analysis_max_edge",
        "cross_anchor_montage_full_page_max_edge",
        "cross_anchor_montage_tile_edge",
        "cross_anchor_montage_columns",
        "cross_anchor_llm2_batch_size",
        "cross_anchor_llm1_verification_runs",
        "cross_anchor_llm2_localization_runs",
    }
    for name in positive_integer_fields:
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(config["cross_anchor_llm2_localization_runs"]) > 2:
        raise ValueError("cross_anchor_llm2_localization_runs must be 1 or 2")
    boolean_fields = {
        "cross_anchor_retain_uncertain_candidates",
        "cross_anchor_retain_rejected_candidates",
        "cross_anchor_retain_uncertain_fallback_candidates",
        "cross_anchor_fallback_generates_anchors",
    }
    for name in boolean_fields:
        if not isinstance(config[name], bool):
            raise ValueError(f"{name} must be boolean")
    for name in ("red_min_channel", "red_min_excess"):
        if not 0 <= int(config[name]) <= 255:
            raise ValueError(f"{name} must be between 0 and 255")
    ratio_fields = required - {
        *positive_integer_fields,
        "red_min_channel",
        "red_min_excess",
        *boolean_fields,
    }
    for name in ratio_fields:
        if not 0 <= float(config[name]) <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if float(config["arm_inner_radius_ratio"]) >= float(
        config["arm_outer_radius_ratio"]
    ):
        raise ValueError("arm inner radius must be smaller than outer radius")

    truth_payload = json.loads(truth_path.read_text(encoding="utf-8"))
    pages = truth_payload.get("pages")
    if not isinstance(pages, dict):
        raise ValueError("truth JSON must contain a pages object")
    truth_by_label = {}
    for label in labels:
        page = pages.get(label)
        if not isinstance(page, dict):
            raise ValueError(f"missing truth page: {label}")
        regions = page.get("regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"truth page has no regions: {label}")
        seen_truth_ids = set()
        for region in regions:
            truth_id = region.get("truth_id")
            if not isinstance(truth_id, str) or not truth_id:
                raise ValueError(f"invalid truth_id for page: {label}")
            if truth_id in seen_truth_ids:
                raise ValueError(f"duplicate truth_id for page {label}: {truth_id}")
            seen_truth_ids.add(truth_id)
            validate_normalized_bbox(region.get("source_bbox_normalized"))
        truth_by_label[label] = regions
    return config, truth_by_label


def _write_report(output_dir: Path, summaries: list[dict]) -> None:
    lines = [
        "# 视觉识别流程诊断报告",
        "",
        "> CV 组件/证据组数量不等于错题数量；本表只用于定位首次数量偏差。",
        "",
        "| 图片 | 人工错题 | CV组件 | CV证据组 | 当前primitive | 当前事件 | 当前定位 | 当前内容 | 当前真值命中 | 当前真值召回 | 当前首次偏差 | 实验primitive | 稳定事件 | 重复event候选 | 重复primitive候选 | 跨单元圈叉候选 | 圈叉归属异常 | 未分配primitive | 未覆盖CV组件 | LLM实验耗时(ms) | 新方案CV候选 | 新方案确认红叉 | 保留uncertain | 保留高分CV | LLM漏检补充 | 复核通过fallback | 独立扫描红叉 | 独立扫描支持 | 本地几何合并 | LLM1真值召回 | LLM1区域外红叉 | LLM1真值重复 | 新方案错题定位 | 几何异常 | 重复错题候选 | 真值重复归属 | 新方案真值命中 | 新方案真值召回 | 内容/OCR状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        item = summary["checkpoints"]
        experiment_ms = None
        if item["stable_mark_detection_ms"] is not None:
            experiment_ms = round(
                item["stable_mark_detection_ms"]
                + (item["stable_event_consolidation_ms"] or 0),
                2,
            )
        divergence = (
            "未运行"
            if summary["pipeline_status"] == "not_run"
            else summary["first_count_divergence"] or "无数量偏差"
        )
        lines.append(
            "| {label} | {expected} | {components} | {groups} | {current_primitives} | {marks} | {located} | {content} | {pipeline_truth_matched} | {pipeline_truth_recall} | {divergence} | {primitives} | {stable} | {duplicate_events} | {duplicate_primitives} | {geometry_candidates} | {membership_violations} | {unassigned} | {uncovered} | {experiment_ms} | {cross_candidates} | {confirmed_crosses} | {uncertain_retained} | {high_score_retained} | {fallback_crosses} | {fallback_verified} | {independent_scan} | {independent_supported} | {local_merges} | {llm1_truth_recall} | {llm1_false_crosses} | {llm1_duplicate_truth} | {anchored_questions} | {anchored_geometry} | {duplicate_questions} | {duplicate_truth} | {truth_matched} | {truth_recall} | {content_ocr_status} |".format(
                label=summary["label"],
                expected=item["expected_error_count"],
                components=item["cv_raw_component_count"],
                groups=item["cv_evidence_group_count"],
                current_primitives=item["llm_mark_primitive_count"],
                marks=item["normalized_mark_event_count"],
                located=item["localized_mark_count"],
                content=item["content_item_count"],
                pipeline_truth_matched=item["pipeline_truth_matched_count"],
                pipeline_truth_recall=item["pipeline_truth_recall"],
                divergence=divergence,
                primitives=item["stable_primitive_count"],
                stable=item["stable_event_count"],
                duplicate_events=item["stable_duplicate_event_candidate_count"],
                duplicate_primitives=item[
                    "stable_duplicate_primitive_candidate_count"
                ],
                geometry_candidates=item[
                    "stable_cross_circle_geometry_candidate_count"
                ],
                membership_violations=item[
                    "stable_primitive_membership_violation_count"
                ],
                unassigned=item["stable_unassigned_primitive_count"],
                uncovered=item["stable_uncovered_component_count"],
                experiment_ms=experiment_ms,
                cross_candidates=item["cross_anchor_cv_candidate_count"],
                confirmed_crosses=item["cross_anchor_confirmed_cross_count"],
                uncertain_retained=item[
                    "cross_anchor_uncertain_retained_count"
                ],
                high_score_retained=item[
                    "cross_anchor_high_score_retained_count"
                ],
                fallback_crosses=item["cross_anchor_fallback_cross_count"],
                fallback_verified=item["cross_anchor_fallback_verified_count"],
                independent_scan=item["cross_anchor_independent_scan_count"],
                independent_supported=item[
                    "cross_anchor_independent_supported_count"
                ],
                local_merges=item["cross_anchor_local_geometry_merge_count"],
                llm1_truth_recall=item["cross_anchor_llm1_truth_recall"],
                llm1_false_crosses=item[
                    "cross_anchor_llm1_false_cross_count"
                ],
                llm1_duplicate_truth=item[
                    "cross_anchor_llm1_duplicate_truth_candidate_count"
                ],
                anchored_questions=item["cross_anchor_matched_question_count"],
                anchored_geometry=item["cross_anchor_geometry_violation_count"],
                duplicate_questions=item[
                    "cross_anchor_duplicate_question_candidate_count"
                ],
                duplicate_truth=item[
                    "cross_anchor_duplicate_truth_candidate_count"
                ],
                truth_matched=item["cross_anchor_truth_matched_count"],
                truth_recall=item["cross_anchor_truth_recall"],
                content_ocr_status=item["cross_anchor_content_ocr_status"],
            )
        )
    lines.extend(
        [
            "",
            "## 召回优先新方案判定",
            "",
            "> 稳定错题事件由LLM2原始错题框做审计性聚类得到，不改写模型原始返回；目标是稳定真值召回为1.0，允许存在误报事件。",
            "",
            "| 图片 | LLM1核验次数 | 不稳定CV候选 | 保留LLM1拒绝CV | fallback uncertain审计 | fallback生成锚点 | LLM2定位次数 | 定向复查触发 | 定向复查请求 | 第一次真值召回 | 第一次最小真值覆盖 | 定向复查新增找回 | 定向复查新增误报 | 稳定错题事件 | 合并真值命中 | 合并真值召回 | 合并最小真值覆盖 | 合并误报事件 | 新方案LLM请求 |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        item = summary["checkpoints"]
        lines.append(
            "| {label} | {runs} | {unstable} | {rejected} | {fallback_uncertain} | {fallback_generates} | {llm2_runs} | {retry_triggers} | {retry_requests} | {first_recall} | {first_coverage} | {recovered} | {additional_false} | {events} | {matched} | {recall} | {union_coverage} | {false_events} | {requests} |".format(
                label=summary["label"],
                runs=item["cross_anchor_llm1_verification_run_count"],
                unstable=item["cross_anchor_llm1_unstable_candidate_count"],
                rejected=item["cross_anchor_rejected_retained_count"],
                fallback_uncertain=item[
                    "cross_anchor_fallback_uncertain_retained_count"
                ],
                fallback_generates=item[
                    "cross_anchor_fallback_generates_anchors"
                ],
                llm2_runs=item["cross_anchor_llm2_localization_run_count"],
                retry_triggers=item["cross_anchor_llm2_retry_trigger_count"],
                retry_requests=item["cross_anchor_llm2_retry_request_count"],
                first_recall=item[
                    "cross_anchor_first_pass_stable_truth_recall"
                ],
                first_coverage=item[
                    "cross_anchor_first_pass_minimum_truth_coverage"
                ],
                recovered=item[
                    "cross_anchor_second_pass_recovered_truth_count"
                ],
                additional_false=item[
                    "cross_anchor_second_pass_additional_false_event_count"
                ],
                events=item["cross_anchor_stable_question_event_count"],
                matched=item["cross_anchor_stable_truth_matched_count"],
                recall=item["cross_anchor_stable_truth_recall"],
                union_coverage=item[
                    "cross_anchor_stable_minimum_truth_coverage"
                ],
                false_events=item["cross_anchor_stable_false_event_count"],
                requests=item["cross_anchor_llm_request_count"],
            )
        )
    (output_dir / "comparison-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_timing_report(output_dir: Path, summaries: list[dict]) -> None:
    lines = [
        "# 视觉诊断耗时对比",
        "",
        "> 单位均为毫秒。旧生产流程、旧stable实验和新cross-anchor实验在同一轮中串行执行；整页总耗时包含本地CV、文件输出及所有启用实验。",
        "",
        "| 图片 | 整页总耗时 | 红色证据CV | 红叉候选CV | 旧生产流程 | 旧stable实验 | 新方案总耗时 | LLM1核验次数 | 新方案LLM请求 | 定向复查触发 | 定向复查请求 | LLM1候选核验 | 独立漏检扫描 | fallback复核 | LLM2定位总计 | 定向复查耗时 | 后置审计 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        case_timings = summary.get("timings_ms") or {}
        checkpoints = summary["checkpoints"]
        stage_timings = checkpoints.get("cross_anchor_stage_timings_ms") or {}
        lines.append(
            "| {label} | {total} | {red_cv} | {cross_cv} | {production} | {stable} | {cross_anchor} | {runs} | {requests} | {retry_triggers} | {retry_requests} | {verify} | {scan} | {fallback} | {llm2} | {llm2_second} | {audit} |".format(
                label=summary["label"],
                total=case_timings.get("total"),
                red_cv=case_timings.get("red_evidence_cv"),
                cross_cv=case_timings.get("cross_candidate_cv"),
                production=case_timings.get("production_pipeline"),
                stable=case_timings.get("stable_event_experiment"),
                cross_anchor=case_timings.get("cross_anchor_experiment"),
                runs=checkpoints.get(
                    "cross_anchor_llm1_verification_run_count"
                ),
                requests=checkpoints.get("cross_anchor_llm_request_count"),
                retry_triggers=checkpoints.get(
                    "cross_anchor_llm2_retry_trigger_count"
                ),
                retry_requests=checkpoints.get(
                    "cross_anchor_llm2_retry_request_count"
                ),
                verify=stage_timings.get("llm1_candidate_verification"),
                scan=stage_timings.get("independent_cross_scan"),
                fallback=stage_timings.get(
                    "fallback_montage_and_verification"
                ),
                llm2=stage_timings.get("llm2_localization"),
                llm2_second=stage_timings.get(
                    "llm2_localization_run_002"
                ),
                audit=stage_timings.get("post_llm2_audit"),
            )
        )
    (output_dir / "timing-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture CV and each three-stage MiniMax exchange for diagnosis."
    )
    parser.add_argument("images", nargs="+", help="label=absolute-image-path")
    parser.add_argument("--expected", action="append", default=[], help="label=count")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", default="chinese")
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument(
        "--cv-cross-only",
        action="store_true",
        help="run deterministic local red-cross CV and truth comparison without LLM",
    )
    parser.add_argument(
        "--cross-cv-config",
        help="JSON configuration for --cv-cross-only or --compare-cross-anchor",
    )
    parser.add_argument(
        "--truth-regions",
        help="truth-regions.json used by CV and cross-anchor comparison",
    )
    parser.add_argument(
        "--compare-stable-events",
        action="store_true",
        help=(
            "also run independent full-page mark detection, stable event consolidation, "
            "and audit-only CV validation"
        ),
    )
    parser.add_argument(
        "--compare-cross-anchor",
        action="store_true",
        help=(
            "also run CV candidate verification, independently reverify fallback "
            "crosses, and batch cross-anchored question localization"
        ),
    )
    parser.add_argument(
        "--stable-primitive-duplicate-containment-threshold",
        type=float,
        help="same-type primitive containment ratio used by diagnostic audit",
    )
    parser.add_argument(
        "--stable-cross-circle-max-center-distance",
        type=float,
        help="maximum normalized circle/cross center distance used by diagnostic audit",
    )
    args = parser.parse_args()

    if args.cv_cross_only:
        if not args.cross_cv_config:
            parser.error("--cv-cross-only requires --cross-cv-config")
        if not args.truth_regions:
            parser.error("--cv-cross-only requires --truth-regions")
        if args.compare_stable_events:
            parser.error("--cv-cross-only cannot be combined with --compare-stable-events")
        if args.compare_cross_anchor:
            parser.error("--cv-cross-only cannot be combined with --compare-cross-anchor")

    if args.compare_cross_anchor:
        if not args.cross_cv_config:
            parser.error("--compare-cross-anchor requires --cross-cv-config")
        if not args.truth_regions:
            parser.error("--compare-cross-anchor requires --truth-regions")

    if args.compare_stable_events:
        if args.stable_primitive_duplicate_containment_threshold is None:
            parser.error(
                "--compare-stable-events requires "
                "--stable-primitive-duplicate-containment-threshold"
            )
        if not 0 <= args.stable_primitive_duplicate_containment_threshold <= 1:
            parser.error(
                "--stable-primitive-duplicate-containment-threshold must be between 0 and 1"
            )
        if args.stable_cross_circle_max_center_distance is None:
            parser.error(
                "--compare-stable-events requires "
                "--stable-cross-circle-max-center-distance"
            )
        if args.stable_cross_circle_max_center_distance <= 0:
            parser.error("--stable-cross-circle-max-center-distance must be positive")

    try:
        images = _parse_labeled_paths(args.images)
        expected = _parse_expected(args.expected)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    cross_cv_config = None
    truth_by_label = {}
    if args.cv_cross_only or args.compare_cross_anchor:
        try:
            cross_cv_config, truth_by_label = load_cross_cv_inputs(
                Path(args.cross_cv_config).expanduser().resolve(),
                Path(args.truth_regions).expanduser().resolve(),
                [label for label, _path in images],
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        output_dir / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "warning": "Contains worksheet images, prompts, and model responses; delete after analysis.",
            "cv_only": args.cv_only or args.cv_cross_only,
            "cv_cross_only": args.cv_cross_only,
            "compare_stable_events": args.compare_stable_events,
            "compare_cross_anchor": args.compare_cross_anchor,
            "stable_primitive_duplicate_containment_threshold": (
                args.stable_primitive_duplicate_containment_threshold
            ),
            "stable_cross_circle_max_center_distance": (
                args.stable_cross_circle_max_center_distance
            ),
            "labels": [label for label, _path in images],
        },
    )
    summaries = [
        run_case(
            label=label,
            image_path=path,
            output_dir=output_dir,
            expected_count=expected.get(label),
            subject_hint=args.subject,
            cv_only=args.cv_only or args.cv_cross_only,
            cross_cv_config=cross_cv_config,
            truth_regions=truth_by_label.get(label),
            compare_stable_events=args.compare_stable_events,
            compare_cross_anchor=args.compare_cross_anchor,
            primitive_duplicate_containment_threshold=(
                args.stable_primitive_duplicate_containment_threshold
            ),
            cross_circle_max_center_distance=(
                args.stable_cross_circle_max_center_distance
            ),
        )
        for label, path in images
    ]
    _write_json(output_dir / "summary.json", summaries)
    _write_report(output_dir, summaries)
    _write_timing_report(output_dir, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return (
        1
        if any(
            summary["error"]
            or summary["stable_experiment_error"]
            or summary["cross_anchor_experiment_error"]
            for summary in summaries
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
