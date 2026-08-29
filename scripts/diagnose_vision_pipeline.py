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
    stable_experiment: dict | None = None,
) -> dict:
    pipeline_ran = pipeline is not None
    stable_experiment_ran = stable_experiment is not None
    pipeline = pipeline or {}
    checkpoints = {
        "expected_error_count": expected_count,
        "cv_raw_component_count": cv.get("raw_component_count"),
        "cv_evidence_group_count": cv.get("evidence_group_count"),
        "llm_mark_primitive_count": pipeline.get("mark_primitive_count"),
        "normalized_mark_event_count": pipeline.get("mark_event_count"),
        "localized_mark_count": pipeline.get("localized_mark_count"),
        "content_item_count": pipeline.get("content_item_count"),
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
    compare_stable_events: bool = False,
    primitive_duplicate_containment_threshold: float | None = None,
    cross_circle_max_center_distance: float | None = None,
) -> dict:
    case_dir = output_dir / label
    case_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(image_path, case_dir / ("source" + image_path.suffix.lower()))
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
    pipeline_diagnostic = None
    stable_experiment = None
    stable_experiment_error = None
    error = None
    if not cv_only:
        recorder = ExchangeRecorder(case_dir)
        client = MiniMaxVisionClient.from_settings()
        client.diagnostic_event_sink = recorder
        recording_client = RecordingVisionClient(client, recorder)
        arguments = _pipeline_arguments()
        _write_json(case_dir / "effective-config.json", arguments)
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
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "message": str(exc),
                "diagnostic": getattr(exc, "diagnostic", None),
            }
            pipeline_diagnostic = getattr(exc, "diagnostic", None)
            _write_json(case_dir / "pipeline-error.json", error)
        if compare_stable_events:
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
    summary = build_summary(
        label=label,
        expected_count=expected_count,
        cv=cv,
        pipeline=pipeline_diagnostic,
        stable_experiment=stable_experiment,
    )
    summary["error"] = error
    summary["stable_experiment_error"] = stable_experiment_error
    _write_json(case_dir / "summary.json", summary)
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


def _write_report(output_dir: Path, summaries: list[dict]) -> None:
    lines = [
        "# 视觉识别流程诊断报告",
        "",
        "> CV 组件/证据组数量不等于错题数量；本表只用于定位首次数量偏差。",
        "",
        "| 图片 | 人工错题 | CV组件 | CV证据组 | 当前primitive | 当前事件 | 当前定位 | 当前内容 | 当前首次偏差 | 实验primitive | 稳定事件 | 重复event候选 | 重复primitive候选 | 跨单元圈叉候选 | 圈叉归属异常 | 未分配primitive | 未覆盖CV组件 | LLM实验耗时(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            "| {label} | {expected} | {components} | {groups} | {current_primitives} | {marks} | {located} | {content} | {divergence} | {primitives} | {stable} | {duplicate_events} | {duplicate_primitives} | {geometry_candidates} | {membership_violations} | {unassigned} | {uncovered} | {experiment_ms} |".format(
                label=summary["label"],
                expected=item["expected_error_count"],
                components=item["cv_raw_component_count"],
                groups=item["cv_evidence_group_count"],
                current_primitives=item["llm_mark_primitive_count"],
                marks=item["normalized_mark_event_count"],
                located=item["localized_mark_count"],
                content=item["content_item_count"],
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
            )
        )
    (output_dir / "comparison-report.md").write_text(
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
        "--compare-stable-events",
        action="store_true",
        help=(
            "also run independent full-page mark detection, stable event consolidation, "
            "and audit-only CV validation"
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
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        output_dir / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "warning": "Contains worksheet images, prompts, and model responses; delete after analysis.",
            "cv_only": args.cv_only,
            "compare_stable_events": args.compare_stable_events,
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
            cv_only=args.cv_only,
            compare_stable_events=args.compare_stable_events,
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
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return (
        1
        if any(
            summary["error"] or summary["stable_experiment_error"]
            for summary in summaries
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
