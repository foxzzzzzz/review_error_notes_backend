"""Capture CV and three-stage vision evidence without persisting questions.

The output contains worksheet images and model responses. Store it only in a
restricted diagnostic directory and delete it after analysis.
"""

from __future__ import annotations

import argparse
import hashlib
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


CROSS_ANCHOR_PROFILES = (
    "baseline",
    "independent-rescue",
    "anchor-preserving",
    "spatial-grouped",
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


CROSS_ANCHORED_QUESTION_PROMPT = """你是小学作业红叉锚定错题区域定位器。图片可能是叠加蓝色 C0、C1... 编号框的原始整页作业，也可能是不叠加任何蓝框的单个可疑红叉附近局部原图；输入 JSON 给出每个 cross_id 在当前输入图片坐标系中的位置和来源风险。

本阶段唯一目标：逐个判断红叉候选能否关联到一个明确的最小独立作答单元，并只返回完整 question_bbox。不要识别或返回题目文字、学生答案、正确答案、题型、标签或难度。

要求：
1. 每个输入 cross_id 必须且只能返回一次，不得新增、删除、合并或重排。输入可能含误报；若看不到明确老师判错红叉，或无法唯一关联作答单元，必须返回 matched=false，并简要填写 unmatched_reason。
2. matched=true 时，question_bbox 覆盖与该红叉直接相关的完整最小作答单元：印刷提示、学生实际作答以及相关批改痕迹；红叉可以位于框内，也可以紧邻该作答单元，不得因红叉画在作答框上方、右侧或外沿而返回 unmatched；不得吞入相邻兄弟小题或整页区域。
3. question_bbox 不得直接复制红叉框。红叉框只标出批改痕迹，question_bbox 必须扩展到完整印刷提示和学生作答；如果无法看清完整边界，返回 matched=false。
4. 红叉是决定性主锚点。若附近存在老师画出的闭合或近似闭合红圈，可将红圈作为辅助证据，用于确认对应的学生作答、区分相邻作答单元，并确定 question_bbox 的扩展方向和边界。不得仅凭红圈或教师批注新增错题，不得把红圈关联到相邻小题；无需单独返回红圈或教师批注。
5. source=llm_fallback 表示低可信独立扫描结果，不得默认其为真实红叉，必须根据图片重新核验。
6. matched=false 时 question_bbox 必须为 null；matched=true 时 unmatched_reason 应为 null。
7. question_bbox 使用当前输入图片的归一化 [left, top, right, bottom]。若 anchor 的 image_scope=local_retry_crop，当前图片是不带蓝框的局部原图，必须结合输入 JSON 中的 bbox 查找候选，并按当前局部图返回坐标。整页图片上的蓝色编号框只是提示，不是图片原有内容。
8. 只返回严格 JSON，不要解释或 Markdown。

返回格式：{"items":[{"cross_id":0,"matched":true,"question_bbox":[0.1,0.2,0.4,0.5],"unmatched_reason":null,"confidence":0.95},{"cross_id":1,"matched":false,"question_bbox":null,"unmatched_reason":"不是明确红叉","confidence":0.8}]}。

输入 cross anchors：__CROSSES__
"""


SPATIAL_CROSS_GROUP_PROMPT = """你是小学作业红叉锚定错题空间归组器。当前图片是原始作业的一个局部裁图，蓝色 C 编号框和输入 JSON 标出需要核验的红叉锚点。

目标：把每个真实红叉归属到当前裁图中的最小独立错题作答单元；同一道题存在多个红叉时，用一个 group 返回全部对应 cross_ids。

要求：
1. 每个输入 cross_id 必须且只能出现一次：要么属于一个 groups[].cross_ids，要么属于 unmatched；不得新增、遗漏、重复或跨 group 使用同一 ID。
2. question_bbox 覆盖完整最小作答单元，可参考红叉附近红圈来确认边界，但红圈和教师批注不能单独创建错题。
3. 相邻兄弟题必须拆分；只有明确属于同一作答单元的多个红叉才能放入同一 group。
4. 无法确认是红叉或无法唯一定位作答单元时放入 unmatched，不得猜测。
5. question_bbox 使用当前局部裁图归一化坐标，不得返回整页坐标。
6. group_id 从0开始连续。只返回严格 JSON，不要解释或 Markdown。

返回格式：{"groups":[{"group_id":0,"cross_ids":[0,1],"question_bbox":[0.1,0.2,0.8,0.7],"confidence":0.9}],"unmatched":[{"cross_id":2,"reason":"不是明确红叉","confidence":0.8}]}。

输入 anchors：__CROSSES__
"""


HOLISTIC_CROSS_GEOMETRY_PROMPT_V3 = """你是“中国小学作业红叉与错题区域几何识别器”。只处理几何位置，不识别题目文字和答案。

任务：检查完整作业页面，清点所有明确的老师红叉，并将红叉归属到最小但完整的独立答题单元。

规则：
1. 只有能看到两条红色相交笔画、且表示老师判错的 × 才是 cross_anchor。
2. 红圈只能辅助判断红叉归属和答题单元边界；只有红圈没有红叉时不得生成 question_event。
3. 红叉可以没有红圈，仍必须输出。
4. 对勾、下划线、方格线、页码、装饰线、箭头、批注文字和普通圈画不是红叉。
5. 先从上到下、从左到右清点整页红叉，再进行事件聚合；检查页面顶部、底部和边缘。
6. 一个 cross_anchor 表示一个可见红叉。同一答题单元有多个红叉时，全部锚点归入同一个 question_event。
7. 相邻但独立的小题必须生成不同事件，不得因为位置接近而合并。
8. question_bbox 覆盖最小但完整的题干提示、学生作答和相关批改痕迹；不得只框红叉，也不得吞入相邻兄弟题。
9. 每个锚点必须且只能归入一个事件；确实无法定位所属答题单元时，放入 unassigned_cross_anchor_ids，禁止静默丢失。
10. bbox 使用完整输入图的归一化 [left, top, right, bottom]。
11. 只返回指定结构的合法 JSON，不输出解释、Markdown、内容识别字段或额外字段。

输出格式：
{"page_rule":"red_cross_primary","observed_cross_count":1,"cross_anchors":[{"anchor_id":"a0","bbox":[0.1,0.2,0.2,0.3],"visual_evidence":"two_intersecting_red_diagonal_strokes","model_confidence":0.9}],"question_events":[{"event_id":"q0","anchor_ids":["a0"],"question_bbox":[0.05,0.15,0.4,0.35],"model_confidence":0.9}],"circle_only_regions":[],"unassigned_cross_anchor_ids":[]}
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


class SpatialQuestionGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    group_id: int = Field(ge=0)
    cross_ids: list[int] = Field(min_length=1)
    question_bbox: list[float]
    confidence: float = Field(ge=0, le=1)

    @field_validator("question_bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        return validate_normalized_bbox(value)


class SpatialUnmatchedCross(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cross_id: int = Field(ge=0)
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class SpatialQuestionGroupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    groups: list[SpatialQuestionGroup] = Field(default_factory=list)
    unmatched: list[SpatialUnmatchedCross] = Field(default_factory=list)


class HolisticCrossAnchor(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, protected_namespaces=()
    )

    anchor_id: str = Field(min_length=1)
    bbox: list[float]
    visual_evidence: Literal["two_intersecting_red_diagonal_strokes"]
    model_confidence: float = Field(ge=0, le=1)

    @field_validator("bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        return validate_normalized_bbox(value)


class HolisticQuestionEvent(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, protected_namespaces=()
    )

    event_id: str = Field(min_length=1)
    anchor_ids: list[str] = Field(min_length=1)
    question_bbox: list[float]
    model_confidence: float = Field(ge=0, le=1)

    @field_validator("question_bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        return validate_normalized_bbox(value)


class HolisticCircleOnlyRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    bbox: list[float]
    reason: Literal["red_circle_without_cross"]

    @field_validator("bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        return validate_normalized_bbox(value)


class HolisticCrossGeometryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    page_rule: Literal["red_cross_primary"]
    observed_cross_count: int = Field(ge=0)
    cross_anchors: list[HolisticCrossAnchor] = Field(default_factory=list)
    question_events: list[HolisticQuestionEvent] = Field(default_factory=list)
    circle_only_regions: list[HolisticCircleOnlyRegion] = Field(default_factory=list)
    unassigned_cross_anchor_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def anchor_membership_must_be_complete_and_unique(self):
        anchor_ids = [anchor.anchor_id for anchor in self.cross_anchors]
        if self.observed_cross_count != len(anchor_ids):
            raise ValueError("observed cross count must equal cross anchor count")
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("cross anchor ids must be unique")
        event_ids = [event.event_id for event in self.question_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("question event ids must be unique")
        assigned = [
            anchor_id
            for event in self.question_events
            for anchor_id in event.anchor_ids
        ]
        unknown = sorted(set(assigned) - set(anchor_ids))
        if unknown:
            raise ValueError(f"question events reference unknown anchors: {unknown}")
        duplicates = sorted(
            anchor_id for anchor_id in set(assigned) if assigned.count(anchor_id) > 1
        )
        if duplicates:
            raise ValueError(f"cross anchors assigned more than once: {duplicates}")
        unassigned = self.unassigned_cross_anchor_ids
        if len(unassigned) != len(set(unassigned)):
            raise ValueError("unassigned cross anchor ids must be unique")
        unknown_unassigned = sorted(set(unassigned) - set(anchor_ids))
        if unknown_unassigned:
            raise ValueError(
                f"unassigned list references unknown anchors: {unknown_unassigned}"
            )
        overlap = sorted(set(assigned) & set(unassigned))
        if overlap:
            raise ValueError(f"cross anchors both assigned and unassigned: {overlap}")
        missing = sorted(set(anchor_ids) - set(assigned) - set(unassigned))
        if missing:
            raise ValueError(
                f"cross anchors neither assigned nor unassigned: {missing}"
            )
        return self


CrossCandidateVerificationResult.model_rebuild(_types_namespace=globals())
IndependentCrossScanResult.model_rebuild(_types_namespace=globals())
CrossAnchoredQuestionResult.model_rebuild(_types_namespace=globals())
SpatialQuestionGroupResult.model_rebuild(_types_namespace=globals())
HolisticCrossGeometryResult.model_rebuild(_types_namespace=globals())


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


def mark_existing_anchor_scan_support(
    anchors: list[dict],
    independent_scan: IndependentCrossScanResult,
    *,
    min_iou: float,
    max_center_distance_ratio: float,
) -> None:
    for anchor in anchors:
        if any(
            _bbox_iou(anchor["bbox"], cross.bbox) >= min_iou
            or _bbox_center_distance(anchor["bbox"], cross.bbox)
            <= max_center_distance_ratio
            for cross in independent_scan.crosses
        ):
            anchor["independent_scan_supported"] = True


def measure_bbox_red_support(
    image_path: Path,
    bbox: list[float],
    config: dict,
) -> dict:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    left, top, right, bottom = _pixel_bbox(bbox, image.width, image.height)
    left = max(0, min(left, image.width - 1))
    top = max(0, min(top, image.height - 1))
    right = max(left + 1, min(right, image.width))
    bottom = max(top + 1, min(bottom, image.height))
    pixels = np.asarray(image, dtype=np.int16)[top:bottom, left:right]
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    mask = (red >= int(config["red_min_channel"])) & (
        red - np.maximum(green, blue) >= int(config["red_min_excess"])
    )
    return {
        "red_pixel_count": int(mask.sum()),
        "pixel_count": int(mask.size),
        "red_pixel_ratio": round(float(mask.mean()), 6),
    }


def load_cross_anchor_llm1_snapshot(snapshot_dir: Path) -> dict:
    snapshot_dir = snapshot_dir.resolve()
    filenames = (
        "llm1-candidate-verification.json",
        "llm1-independent-scan.json",
        "llm1-fallback-candidate-verification.json",
    )
    payloads = {}
    hashes = {}
    for filename in filenames:
        path = snapshot_dir / filename
        raw = path.read_bytes()
        payloads[filename] = json.loads(raw.decode("utf-8"))
        hashes[filename] = hashlib.sha256(raw).hexdigest()
    return {
        "source": str(snapshot_dir),
        "sha256": hashes,
        "verification": CrossCandidateVerificationResult.model_validate(
            payloads["llm1-candidate-verification.json"]
        ),
        "independent_scan": IndependentCrossScanResult.model_validate(
            payloads["llm1-independent-scan.json"]
        ),
        "fallback_verification": CrossCandidateVerificationResult.model_validate(
            payloads["llm1-fallback-candidate-verification.json"]
        ),
    }


def resolve_cross_anchor_snapshot_dir(root: Path, label: str) -> Path:
    candidates = (
        root / label / "cross-anchor-experiment",
        root / "pages" / label / "cross-anchor-experiment",
    )
    matches = [candidate.resolve() for candidate in candidates if candidate.is_dir()]
    if len(matches) != 1:
        raise ValueError(
            f"expected one cross-anchor snapshot directory for {label} under {root}"
        )
    return matches[0]


def select_independent_rescue_crosses(
    *,
    existing_anchors: list[dict],
    independent_scan: IndependentCrossScanResult,
    fallback_verification: CrossCandidateVerificationResult,
    image_path: Path,
    config: dict,
) -> tuple[IndependentCrossScanResult, dict]:
    verdict_by_id = {
        verdict.candidate_id: verdict for verdict in fallback_verification.verdicts
    }
    rescued = []
    entries = []
    supported_existing_count = 0
    for scan_index, cross in enumerate(independent_scan.crosses):
        matching_anchor = next(
            (
                anchor
                for anchor in existing_anchors
                if _bbox_iou(anchor["bbox"], cross.bbox)
                >= float(config["cross_anchor_fallback_merge_iou_threshold"])
                or _bbox_center_distance(anchor["bbox"], cross.bbox)
                <= float(
                    config["cross_anchor_fallback_merge_center_distance_ratio"]
                )
            ),
            None,
        )
        support = measure_bbox_red_support(image_path, cross.bbox, config)
        area_ratio = _bbox_area(cross.bbox)
        edge_margin = float(config["cross_anchor_rescue_edge_margin_ratio"])
        edge_complete = (
            cross.bbox[0] >= edge_margin
            and cross.bbox[1] >= edge_margin
            and cross.bbox[2] <= 1 - edge_margin
            and cross.bbox[3] <= 1 - edge_margin
        )
        verdict = verdict_by_id.get(scan_index)
        reasons = []
        if cross.confidence < float(config["cross_anchor_rescue_min_scan_confidence"]):
            reasons.append("scan_confidence_below_minimum")
        if support["red_pixel_ratio"] < float(
            config["cross_anchor_rescue_min_red_pixel_ratio"]
        ):
            reasons.append("red_pixel_ratio_below_minimum")
        if area_ratio < float(config["cross_anchor_rescue_min_bbox_area_ratio"]):
            reasons.append("bbox_area_below_minimum")
        if area_ratio > float(config["cross_anchor_rescue_max_bbox_area_ratio"]):
            reasons.append("bbox_area_above_maximum")
        if not edge_complete:
            reasons.append("bbox_touches_page_edge")

        if matching_anchor is not None:
            matching_anchor["independent_scan_supported"] = True
            supported_existing_count += 1
            decision = "supports_existing_anchor"
        elif not reasons:
            rescued.append(cross)
            decision = "independent_scan_rescue"
        else:
            decision = "audit_only"
        entries.append(
            {
                "scan_index": scan_index,
                "bbox": list(cross.bbox),
                "scan_confidence": cross.confidence,
                "fallback_disposition": (
                    verdict.disposition if verdict is not None else None
                ),
                "fallback_confidence": (
                    verdict.confidence if verdict is not None else None
                ),
                **support,
                "bbox_area_ratio": round(area_ratio, 6),
                "edge_complete": edge_complete,
                "matched_cross_id": (
                    matching_anchor["cross_id"] if matching_anchor is not None else None
                ),
                "decision": decision,
                "reasons": reasons,
            }
        )
    return IndependentCrossScanResult(crosses=rescued), {
        "scan_count": len(independent_scan.crosses),
        "rescued_count": len(rescued),
        "supported_existing_count": supported_existing_count,
        "entries": entries,
        "policy": "Truth-blind confidence, red-pixel, area, edge, and geometry gates.",
    }


def build_local_anchor_context_bbox(
    anchor_bbox: list[float],
    config: dict,
) -> list[float]:
    horizontal_padding = float(
        config["cross_anchor_context_horizontal_padding_ratio"]
    )
    vertical_padding = float(config["cross_anchor_context_vertical_padding_ratio"])
    left = max(0.0, anchor_bbox[0] - horizontal_padding)
    top = max(0.0, anchor_bbox[1] - vertical_padding)
    right = min(1.0, anchor_bbox[2] + horizontal_padding)
    bottom = min(1.0, anchor_bbox[3] + vertical_padding)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    width = max(
        right - left,
        float(config["cross_anchor_context_min_width_ratio"]),
    )
    height = max(
        bottom - top,
        float(config["cross_anchor_context_min_height_ratio"]),
    )
    max_area = float(config["cross_anchor_context_max_area_ratio"])
    if width * height > max_area:
        scale = math.sqrt(max_area / (width * height))
        width *= scale
        height *= scale
    left = max(0.0, min(center_x - width / 2, 1 - width))
    top = max(0.0, min(center_y - height / 2, 1 - height))
    return [
        round(left, 6),
        round(top, 6),
        round(left + width, 6),
        round(top + height, 6),
    ]


def build_anchor_preservation_events(
    *,
    anchors: list[dict],
    result: CrossAnchoredQuestionResult,
    geometry_audit: dict,
    batch_errors: list[dict],
    image_path: Path,
    config: dict,
) -> dict:
    item_by_id = {item.cross_id: item for item in result.items}
    geometry_violation_ids = {
        violation["cross_id"] for violation in geometry_audit["violations"]
    }
    failed_ids = {
        cross_id for error in batch_errors for cross_id in error["cross_ids"]
    }
    strong_sources = set(config["cross_anchor_preserve_strong_sources"])
    events = []
    audit = []
    for anchor in sorted(anchors, key=lambda item: item["cross_id"]):
        cross_id = anchor["cross_id"]
        item = item_by_id.get(cross_id)
        red_support = measure_bbox_red_support(image_path, anchor["bbox"], config)
        strong_evidence = (
            (
                anchor["source"] in strong_sources
                or anchor["independent_scan_supported"]
            )
            and float(anchor.get("confidence") or 0)
            >= float(config["cross_anchor_preserve_min_confidence"])
            and red_support["red_pixel_ratio"]
            >= float(config["cross_anchor_preserve_min_red_pixel_ratio"])
        )
        reasons = []
        if cross_id in failed_ids:
            reasons.append("llm2_batch_failed")
        elif item is None:
            reasons.append("llm2_result_missing")
        elif cross_id in geometry_violation_ids:
            reasons.append("llm2_geometry_invalid")
        elif not item.matched:
            reasons.append("llm2_unmatched")

        evidence_tier = "strong" if strong_evidence else "weak"
        if item is not None and item.matched and not reasons:
            status = "confirmed"
            bbox_source = "llm2"
            question_bbox = list(item.question_bbox)
            confidence = item.confidence
        else:
            status = "needs_review"
            bbox_source = "local_anchor_context"
            question_bbox = build_local_anchor_context_bbox(anchor["bbox"], config)
            confidence = float(anchor.get("confidence") or 0)
        events.append(
            {
                "event_id": len(events),
                "cross_ids": [cross_id],
                "representative_cross_id": cross_id,
                "question_bboxes": [question_bbox],
                "confidence": confidence,
                "status": status,
                "evidence_tier": evidence_tier,
                "bbox_source": bbox_source,
            }
        )
        audit.append(
            {
                "cross_id": cross_id,
                "status": status,
                "strong_evidence": strong_evidence,
                "evidence_tier": evidence_tier,
                "source": anchor["source"],
                "independent_scan_supported": anchor["independent_scan_supported"],
                "red_pixel_ratio": red_support["red_pixel_ratio"],
                "reasons": reasons,
            }
        )
    return {
        "event_count": len(events),
        "events": events,
        "audit": audit,
        "silently_dropped_anchor_count": len(anchors) - len(events),
        "policy": (
            "Every anchor is retained. Valid LLM2 regions are confirmed; all "
            "other anchors become local needs_review events with an evidence tier."
        ),
    }


def group_cross_anchors_spatially(
    anchors: list[dict],
    config: dict,
) -> list[dict]:
    row_distance = float(config["cross_anchor_spatial_row_distance_ratio"])
    horizontal_gap = float(config["cross_anchor_spatial_horizontal_gap_ratio"])
    max_anchors = int(config["cross_anchor_spatial_max_anchors_per_group"])
    padding = float(config["cross_anchor_spatial_crop_padding_ratio"])
    max_crop_area = float(config["cross_anchor_spatial_max_crop_area_ratio"])
    ordered = sorted(
        anchors,
        key=lambda item: (
            (item["bbox"][1] + item["bbox"][3]) / 2,
            (item["bbox"][0] + item["bbox"][2]) / 2,
            item["cross_id"],
        ),
    )
    rows = []
    for anchor in ordered:
        center_y = (anchor["bbox"][1] + anchor["bbox"][3]) / 2
        matching_row = next(
            (
                row
                for row in rows
                if abs(center_y - row["center_y"]) <= row_distance
            ),
            None,
        )
        if matching_row is None:
            rows.append({"center_y": center_y, "anchors": [anchor]})
        else:
            matching_row["anchors"].append(anchor)
            matching_row["center_y"] = sum(
                (item["bbox"][1] + item["bbox"][3]) / 2
                for item in matching_row["anchors"]
            ) / len(matching_row["anchors"])

    groups = []
    for row in rows:
        current = []
        for anchor in sorted(
            row["anchors"],
            key=lambda item: (
                (item["bbox"][0] + item["bbox"][2]) / 2,
                item["cross_id"],
            ),
        ):
            proposed = current + [anchor]
            union = list(proposed[0]["bbox"])
            for item in proposed[1:]:
                union = _bbox_union(union, item["bbox"])
            crop_bbox = [
                max(0.0, union[0] - padding),
                max(0.0, union[1] - padding),
                min(1.0, union[2] + padding),
                min(1.0, union[3] + padding),
            ]
            split = bool(current) and (
                len(current) >= max_anchors
                or anchor["bbox"][0] - current[-1]["bbox"][2] > horizontal_gap
                or _bbox_area(crop_bbox) > max_crop_area
            )
            if split:
                groups.append(current)
                current = [anchor]
            else:
                current = proposed
        if current:
            groups.append(current)

    payload = []
    for group_id, group_anchors in enumerate(groups):
        union = list(group_anchors[0]["bbox"])
        for anchor in group_anchors[1:]:
            union = _bbox_union(union, anchor["bbox"])
        crop_bbox = [
            round(max(0.0, union[0] - padding), 6),
            round(max(0.0, union[1] - padding), 6),
            round(min(1.0, union[2] + padding), 6),
            round(min(1.0, union[3] + padding), 6),
        ]
        payload.append(
            {
                "group_id": group_id,
                "crop_bbox": crop_bbox,
                "anchors": group_anchors,
            }
        )
    return payload


def audit_spatial_request_budget(
    *,
    anchor_count: int,
    baseline_batch_size: int,
    spatial_group_count: int,
) -> dict:
    baseline_request_budget = (
        math.ceil(anchor_count / baseline_batch_size) if anchor_count else 0
    )
    eligible = spatial_group_count <= baseline_request_budget
    return {
        "eligible": eligible,
        "anchor_count": anchor_count,
        "baseline_batch_size": baseline_batch_size,
        "baseline_request_budget": baseline_request_budget,
        "spatial_group_count": spatial_group_count,
        "saved_request_count": baseline_request_budget - spatial_group_count,
        "reason": (
            None
            if eligible
            else "spatial_group_count_exceeds_baseline_budget"
        ),
    }


def audit_spatial_group_membership(
    result: SpatialQuestionGroupResult,
    cross_ids: list[int],
) -> dict:
    returned_ids = [
        cross_id for group in result.groups for cross_id in group.cross_ids
    ] + [item.cross_id for item in result.unmatched]
    expected = set(cross_ids)
    returned = set(returned_ids)
    duplicate_ids = sorted(
        cross_id for cross_id in returned if returned_ids.count(cross_id) > 1
    )
    missing_ids = sorted(expected - returned)
    unknown_ids = sorted(returned - expected)
    return {
        "valid": not (duplicate_ids or missing_ids or unknown_ids),
        "input_cross_ids": sorted(cross_ids),
        "missing_cross_ids": missing_ids,
        "duplicate_cross_ids": duplicate_ids,
        "unknown_cross_ids": unknown_ids,
        "policy": "Every input cross_id appears exactly once in groups or unmatched.",
    }


def map_spatial_group_result_to_source(
    result: SpatialQuestionGroupResult,
    source_crop_bbox: list[float],
) -> SpatialQuestionGroupResult:
    return SpatialQuestionGroupResult.model_validate(
        {
            "groups": [
                {
                    **group.model_dump(),
                    "question_bbox": [
                        round(value, 6)
                        for value in _map_bbox_from_crop(
                            group.question_bbox, source_crop_bbox
                        )
                    ],
                }
                for group in result.groups
            ],
            "unmatched": [item.model_dump() for item in result.unmatched],
        }
    )


def write_spatial_anchor_group_crop(
    image_path: Path,
    output_path: Path,
    group: dict,
) -> dict:
    crop_bbox = group["crop_bbox"]
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    pixel_bbox = _pixel_bbox(crop_bbox, image.width, image.height)
    crop = image.crop(pixel_bbox)
    mapped_anchors = [
        {
            **anchor,
            "bbox": [
                round(value, 6)
                for value in _map_bbox_to_crop(anchor["bbox"], crop_bbox)
            ],
            "image_scope": "spatial_group_crop",
        }
        for anchor in group["anchors"]
    ]
    draw = ImageDraw.Draw(crop)
    for anchor in mapped_anchors:
        left, top, right, bottom = _pixel_bbox(
            anchor["bbox"], crop.width, crop.height
        )
        draw.rectangle((left, top, right, bottom), outline="blue", width=2)
        draw.text((left + 2, top + 2), f"C{anchor['cross_id']}", fill="blue")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, format="JPEG", quality=92)
    return {
        "group_id": group["group_id"],
        "source_crop_bbox": crop_bbox,
        "anchors": mapped_anchors,
    }


def run_spatial_group_localization(
    *,
    image_path: Path,
    experiment_dir: Path,
    client,
    groups: list[dict],
    subject_hint: str | None,
) -> dict:
    mapped_groups = []
    unmatched = []
    errors = []
    request_count = 0
    next_group_id = 0
    expected_ids = [
        anchor["cross_id"] for group in groups for anchor in group["anchors"]
    ]
    for group in groups:
        request_count += 1
        group_id = group["group_id"]
        image_output = experiment_dir / f"spatial-group-{group_id:03d}.jpg"
        mapping = write_spatial_anchor_group_crop(image_path, image_output, group)
        _write_json(
            experiment_dir / f"spatial-group-{group_id:03d}-mapping.json",
            mapping,
        )
        group_cross_ids = [anchor["cross_id"] for anchor in group["anchors"]]
        try:
            local_result = client.locate_spatial_cross_groups(
                str(image_output), mapping["anchors"], subject_hint
            )
            _write_json(
                experiment_dir / f"spatial-group-{group_id:03d}-result-local.json",
                local_result,
            )
            membership = audit_spatial_group_membership(
                local_result, group_cross_ids
            )
            _write_json(
                experiment_dir
                / f"spatial-group-{group_id:03d}-membership-audit.json",
                membership,
            )
            if not membership["valid"]:
                raise ValueError("spatial group membership audit failed")
            mapped = map_spatial_group_result_to_source(
                local_result, mapping["source_crop_bbox"]
            )
            for question_group in mapped.groups:
                mapped_groups.append(
                    SpatialQuestionGroup(
                        group_id=next_group_id,
                        cross_ids=question_group.cross_ids,
                        question_bbox=question_group.question_bbox,
                        confidence=question_group.confidence,
                    )
                )
                next_group_id += 1
            unmatched.extend(mapped.unmatched)
        except Exception as exc:
            errors.append(
                {
                    "spatial_group_id": group_id,
                    "cross_ids": group_cross_ids,
                    "type": type(exc).__name__,
                    "code": getattr(exc, "code", None),
                    "message": str(exc),
                }
            )
            unmatched.extend(
                SpatialUnmatchedCross(
                    cross_id=cross_id,
                    reason="spatial LLM2 group request failed",
                    confidence=0.0,
                )
                for cross_id in group_cross_ids
            )
    result = SpatialQuestionGroupResult(groups=mapped_groups, unmatched=unmatched)
    membership_audit = audit_spatial_group_membership(result, expected_ids)
    return {
        "result": result,
        "membership_audit": membership_audit,
        "errors": errors,
        "request_count": request_count,
    }


def build_spatial_question_events(
    spatial_result: SpatialQuestionGroupResult,
    anchor_preservation: dict,
) -> dict:
    preserved_by_cross_id = {
        cross_id: event
        for event in anchor_preservation["events"]
        for cross_id in event["cross_ids"]
    }
    accepted_groups = [
        group
        for group in spatial_result.groups
        if all(
            preserved_by_cross_id[cross_id]["status"] == "confirmed"
            for cross_id in group.cross_ids
        )
    ]
    grouped_cross_ids = {
        cross_id for group in accepted_groups for cross_id in group.cross_ids
    }
    events = [
        {
            "event_id": event_id,
            "cross_ids": sorted(group.cross_ids),
            "representative_cross_id": min(group.cross_ids),
            "question_bboxes": [list(group.question_bbox)],
            "confidence": group.confidence,
            "status": "confirmed",
            "bbox_source": "spatial_llm2",
        }
        for event_id, group in enumerate(accepted_groups)
    ]
    for preserved in anchor_preservation["events"]:
        if any(cross_id in grouped_cross_ids for cross_id in preserved["cross_ids"]):
            continue
        events.append({**preserved, "event_id": len(events)})
    return {
        "event_count": len(events),
        "events": events,
        "policy": (
            "Only geometry-valid model groups are primary; every other anchor "
            "keeps its individual preserved event."
        ),
    }


def deduplicate_spatial_question_events(
    events: list[dict],
    anchors: list[dict],
    config: dict,
) -> dict:
    anchor_by_id = {anchor["cross_id"]: anchor for anchor in anchors}
    parent = list(range(len(events)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    decisions = []
    for first_index, first in enumerate(events):
        for second_index in range(first_index + 1, len(events)):
            second = events[second_index]
            first_bbox = first["question_bboxes"][0]
            second_bbox = second["question_bboxes"][0]
            intersection = _bbox_intersection_area(first_bbox, second_bbox)
            smaller_area = min(_bbox_area(first_bbox), _bbox_area(second_bbox))
            containment = intersection / smaller_area if smaller_area else 0.0
            iou = _bbox_iou(first_bbox, second_bbox)
            anchor_distance = min(
                _bbox_center_distance(
                    anchor_by_id[first_id]["bbox"],
                    anchor_by_id[second_id]["bbox"],
                )
                for first_id in first["cross_ids"]
                for second_id in second["cross_ids"]
            )
            should_merge = (
                iou >= float(config["cross_anchor_spatial_dedupe_iou_threshold"])
                or containment
                >= float(
                    config["cross_anchor_spatial_dedupe_containment_threshold"]
                )
            ) and anchor_distance <= float(
                config["cross_anchor_spatial_dedupe_max_anchor_distance_ratio"]
            )
            if should_merge:
                union(first_index, second_index)
            decisions.append(
                {
                    "event_ids": [first["event_id"], second["event_id"]],
                    "question_bbox_iou": round(iou, 6),
                    "containment": round(containment, 6),
                    "anchor_distance": round(anchor_distance, 6),
                    "decision": "merged" if should_merge else "kept_separate",
                }
            )

    clusters = {}
    for index, event in enumerate(events):
        clusters.setdefault(find(index), []).append(event)
    deduped = []
    for cluster in clusters.values():
        representative = min(
            cluster,
            key=lambda event: (-float(event["confidence"]), event["event_id"]),
        )
        deduped.append(
            {
                "event_id": len(deduped),
                "cross_ids": sorted(
                    {cross_id for event in cluster for cross_id in event["cross_ids"]}
                ),
                "representative_cross_id": representative.get(
                    "representative_cross_id",
                    representative["cross_ids"][0],
                ),
                "question_bboxes": [
                    bbox for event in cluster for bbox in event["question_bboxes"]
                ],
                "confidence": representative["confidence"],
                "status": (
                    "confirmed"
                    if any(event["status"] == "confirmed" for event in cluster)
                    else "needs_review"
                ),
                "bbox_source": representative["bbox_source"],
                "source_event_ids": [event["event_id"] for event in cluster],
            }
        )
    return {
        "event_count": len(deduped),
        "events": deduped,
        "audit": decisions,
        "policy": "High-overlap local merges retain every cross and source bbox.",
    }


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


def _find_spatially_separated_shared_question_anchors(
    first_pass: CrossAnchoredQuestionResult,
    anchors: list[dict],
    *,
    question_iou_threshold: float,
    min_anchor_distance_ratio: float,
    min_group_size: int,
) -> set[int]:
    anchor_by_id = {anchor["cross_id"]: anchor for anchor in anchors}
    matched_items = [
        item
        for item in first_pass.items
        if item.matched
        and item.question_bbox is not None
        and item.cross_id in anchor_by_id
    ]
    duplicate_peers: dict[int, set[int]] = {
        item.cross_id: set() for item in matched_items
    }
    for index, first in enumerate(matched_items):
        for second in matched_items[index + 1 :]:
            if (
                _bbox_iou(first.question_bbox, second.question_bbox)
                < question_iou_threshold
            ):
                continue
            duplicate_peers[first.cross_id].add(second.cross_id)
            duplicate_peers[second.cross_id].add(first.cross_id)

    outliers = set()
    for cross_id, peer_ids in duplicate_peers.items():
        if len(peer_ids) + 1 < min_group_size:
            continue
        nearest_peer_distance = min(
            _bbox_center_distance(
                anchor_by_id[cross_id]["bbox"],
                anchor_by_id[peer_id]["bbox"],
            )
            for peer_id in peer_ids
        )
        if nearest_peer_distance >= min_anchor_distance_ratio:
            outliers.add(cross_id)
    return outliers


def select_llm2_retry_anchors(
    first_pass: CrossAnchoredQuestionResult,
    anchors: list[dict],
    geometry_audit: dict,
    *,
    min_center_gap_ratio: float,
    min_first_pass_question_cross_area_ratio: float,
    max_question_cross_iou: float,
    shared_question_iou_threshold: float,
    shared_question_min_anchor_distance_ratio: float,
    shared_question_min_group_size: int,
    max_retry_requests: int,
) -> dict:
    items_by_id = {item.cross_id: item for item in first_pass.items}
    geometry_reasons = {
        violation["cross_id"]: list(violation["reasons"])
        for violation in geometry_audit["violations"]
    }
    strong_anomaly_reasons = {
        "question_bbox_copies_cross",
        "question_bbox_not_larger_than_cross",
        "shared_question_bbox_for_spatially_separated_anchor",
    }
    shared_question_outlier_ids = _find_spatially_separated_shared_question_anchors(
        first_pass,
        anchors,
        question_iou_threshold=shared_question_iou_threshold,
        min_anchor_distance_ratio=(
            shared_question_min_anchor_distance_ratio
        ),
        min_group_size=shared_question_min_group_size,
    )
    candidates = []
    suppressed = []
    for anchor in anchors:
        cross_id = anchor["cross_id"]
        item = items_by_id[cross_id]
        reasons = []
        center_gap = None
        if item.matched and item.question_bbox is not None:
            cross_bbox = anchor["bbox"]
            cross_area = _bbox_area(cross_bbox)
            question_area = _bbox_area(item.question_bbox)
            question_cross_area_ratio = (
                question_area / cross_area if cross_area else 0.0
            )
            question_cross_iou = _bbox_iou(item.question_bbox, cross_bbox)
            cross_center = (
                (cross_bbox[0] + cross_bbox[2]) / 2,
                (cross_bbox[1] + cross_bbox[3]) / 2,
            )
            center_gap = _point_bbox_gap_distance(cross_center, item.question_bbox)
            if center_gap > min_center_gap_ratio:
                reasons.append("cross_center_outside_question_bbox")
            if (
                question_cross_area_ratio
                < min_first_pass_question_cross_area_ratio
            ):
                reasons.append("question_bbox_not_larger_than_cross")
            if question_cross_iou > max_question_cross_iou:
                reasons.append("question_bbox_copies_cross")
            if cross_id in shared_question_outlier_ids:
                reasons.append(
                    "shared_question_bbox_for_spatially_separated_anchor"
                )
            reasons.extend(geometry_reasons.get(cross_id, []))
        else:
            reasons.append("unmatched_retained_anchor")
        if not reasons:
            continue
        center_gap_only = set(reasons).issubset(
            {
                "cross_center_outside_question_bbox",
                "question_bbox_not_near_cross",
            }
        )
        if center_gap_only:
            suppressed.append(
                {
                    "cross_id": cross_id,
                    "reasons": ["center_gap_without_independent_anomaly"],
                    "candidate_reasons": sorted(set(reasons)),
                }
            )
            continue
        if (
            anchor.get("source") == "cv_rejected_retained"
            and anchor.get("independent_scan_supported") is not True
            and "unmatched_retained_anchor" not in reasons
            and not strong_anomaly_reasons.intersection(reasons)
        ):
            suppressed.append(
                {
                    "cross_id": cross_id,
                    "reasons": [
                        "rejected_anchor_without_independent_scan_support"
                    ],
                    "candidate_reasons": sorted(set(reasons)),
                }
            )
            continue
        reason_set = set(reasons)
        priority = (
            0
            if "unmatched_retained_anchor" in reason_set
            else 1
            if reason_set.intersection(
                {"question_bbox_copies_cross", "question_bbox_not_larger_than_cross"}
            )
            else 2
            if "shared_question_bbox_for_spatially_separated_anchor" in reason_set
            else 3
        )
        source_priority = {
            "cv_confirmed": 0,
            "cv_uncertain": 1,
            "cv_high_score_retained": 2,
            "llm_fallback": 3,
            "cv_rejected_retained": 4,
        }.get(anchor.get("source"), 5)
        candidates.append(
            (
                priority,
                source_priority,
                -float(anchor.get("confidence") or 0),
                cross_id,
                anchor,
                {
                    "cross_id": cross_id,
                    "reasons": sorted(reason_set),
                    "cross_center_gap_ratio": (
                        round(center_gap, 6) if center_gap is not None else None
                    ),
                },
            )
        )
    candidates.sort(key=lambda entry: entry[:4])
    selected_candidates = candidates[:max_retry_requests]
    for *_rank, cross_id, _anchor, trigger in candidates[max_retry_requests:]:
        suppressed.append(
            {
                "cross_id": cross_id,
                "reasons": ["retry_budget_exhausted"],
                "candidate_reasons": trigger["reasons"],
            }
        )
    selected = [entry[4] for entry in selected_candidates]
    triggers = [entry[5] for entry in selected_candidates]
    return {
        "trigger_count": len(triggers),
        "anchors": selected,
        "triggers": triggers,
        "suppressed": suppressed,
        "min_center_gap_ratio": min_center_gap_ratio,
        "min_first_pass_question_cross_area_ratio": (
            min_first_pass_question_cross_area_ratio
        ),
        "max_question_cross_iou": max_question_cross_iou,
        "shared_question_iou_threshold": shared_question_iou_threshold,
        "shared_question_min_anchor_distance_ratio": (
            shared_question_min_anchor_distance_ratio
        ),
        "shared_question_min_group_size": shared_question_min_group_size,
        "max_retry_requests": max_retry_requests,
        "policy": (
            "Only actionable first-pass anomalies receive a local LLM2 retry; "
            "strong LLM2 self-inconsistency may override rejected-anchor "
            "source suppression."
        ),
    }


def decide_llm2_retry_results(
    first_pass: CrossAnchoredQuestionResult,
    retry: CrossAnchoredQuestionResult,
    anchors: list[dict],
    *,
    min_question_cross_area_ratio: float,
    max_question_cross_iou: float,
    duplicate_question_iou_threshold: float,
) -> tuple[CrossAnchoredQuestionResult, dict]:
    anchor_by_id = {anchor["cross_id"]: anchor for anchor in anchors}
    first_pass_bboxes = [
        (item.cross_id, item.question_bbox)
        for item in first_pass.items
        if item.matched and item.question_bbox is not None
    ]
    accepted_items = []
    decisions = []
    for item in retry.items:
        reasons = []
        metrics = {
            "question_cross_area_ratio": None,
            "question_cross_iou": None,
            "max_first_pass_question_iou": None,
        }
        anchor = anchor_by_id.get(item.cross_id)
        if not item.matched or item.question_bbox is None:
            reasons.append("retry_unmatched")
        elif anchor is None:
            reasons.append("unknown_cross_id")
        else:
            cross_bbox = anchor["bbox"]
            cross_area = _bbox_area(cross_bbox)
            question_area = _bbox_area(item.question_bbox)
            area_ratio = question_area / cross_area if cross_area > 0 else 0.0
            cross_iou = _bbox_iou(item.question_bbox, cross_bbox)
            metrics["question_cross_area_ratio"] = round(area_ratio, 6)
            metrics["question_cross_iou"] = round(cross_iou, 6)
            if cross_iou > max_question_cross_iou:
                reasons.append("question_bbox_copies_cross")
            if area_ratio < min_question_cross_area_ratio:
                reasons.append("question_bbox_not_larger_than_cross")
            if not reasons:
                max_first_pass_iou = max(
                    (
                        _bbox_iou(item.question_bbox, bbox)
                        for cross_id, bbox in first_pass_bboxes
                        if cross_id != item.cross_id
                    ),
                    default=0.0,
                )
                metrics["max_first_pass_question_iou"] = round(
                    max_first_pass_iou, 6
                )
                if max_first_pass_iou >= duplicate_question_iou_threshold:
                    reasons.append("duplicates_first_pass_question")
        accepted = not reasons
        if accepted:
            accepted_items.append(item.model_dump())
        decisions.append(
            {
                "cross_id": item.cross_id,
                "accepted": accepted,
                "reasons": sorted(set(reasons)),
                **metrics,
            }
        )
    accepted_result = CrossAnchoredQuestionResult.model_validate(
        {"items": accepted_items}
    )
    return accepted_result, {
        "accepted_count": len(accepted_items),
        "rejected_count": len(decisions) - len(accepted_items),
        "min_question_cross_area_ratio": min_question_cross_area_ratio,
        "max_question_cross_iou": max_question_cross_iou,
        "duplicate_question_iou_threshold": duplicate_question_iou_threshold,
        "decisions": decisions,
        "policy": (
            "Retry output is a challenger: only expanded, non-copying, novel "
            "question regions may join first-pass events."
        ),
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
    matched_items = sorted(
        (
            item
            for item in result.items
            if item.matched and item.question_bbox is not None
        ),
        key=lambda item: item.cross_id,
    )
    events = [
        {
            "event_id": event_id,
            "cross_ids": [item.cross_id],
            "representative_cross_id": item.cross_id,
            "question_bboxes": [item.question_bbox],
            "confidence": item.confidence,
        }
        for event_id, item in enumerate(matched_items)
    ]
    return {
        "min_iou": min_iou,
        "event_count": len(events),
        "unmatched_cross_ids": sorted(
            item.cross_id for item in result.items if not item.matched
        ),
        "events": events,
        "policy": (
            "Each cross_id remains an independent recall-first event; bbox overlap "
            "is reported by duplicate audits and never merges distinct anchors."
        ),
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
        if base_index is not None:
            clusters[base_index].append(observation)
        else:
            base_event_by_cross_id[observation["cross_id"]] = len(clusters)
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
            "Event identity is the cross_id: retry observations always join their "
            "own anchor event and distinct cross_ids never merge."
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


def audit_first_pass_llm2_localization_risk(
    first_pass: CrossAnchoredQuestionResult,
    anchors: list[dict],
    llm1_truth_comparison: dict,
    anchored_truth_comparison: dict,
    geometry_audit: dict,
) -> dict:
    first_pass_by_id = {item.cross_id: item for item in first_pass.items}
    expected_truth_by_id = {
        assignment["candidate_id"]: assignment["truth_id"]
        for assignment in llm1_truth_comparison["assignments"]
    }
    localized_truth_by_id = {
        item["cross_id"]: item["truth_id"]
        for item in anchored_truth_comparison["items"]
    }
    geometry_reasons_by_id = {
        violation["cross_id"]: sorted(set(violation["reasons"]))
        for violation in geometry_audit["violations"]
    }
    items = []
    for anchor in sorted(anchors, key=lambda entry: entry["cross_id"]):
        cross_id = anchor["cross_id"]
        first_item = first_pass_by_id[cross_id]
        expected_truth_id = expected_truth_by_id.get(cross_id)
        localized_truth_id = localized_truth_by_id.get(cross_id)
        if expected_truth_id is not None:
            if not first_item.matched:
                localization_status = "unmatched_true_anchor"
            elif localized_truth_id == expected_truth_id:
                localization_status = "correct_truth"
            else:
                localization_status = "wrong_truth"
        elif first_item.matched:
            localization_status = "matched_false_anchor"
        else:
            localization_status = "unmatched_false_anchor"

        question_cross_area_ratio = None
        question_cross_width_ratio = None
        question_cross_height_ratio = None
        cross_center_gap_ratio = None
        question_cross_gap_ratio = None
        if first_item.matched and first_item.question_bbox is not None:
            cross_bbox = anchor["bbox"]
            question_bbox = first_item.question_bbox
            cross_width = cross_bbox[2] - cross_bbox[0]
            cross_height = cross_bbox[3] - cross_bbox[1]
            cross_area = _bbox_area(cross_bbox)
            cross_center = (
                (cross_bbox[0] + cross_bbox[2]) / 2,
                (cross_bbox[1] + cross_bbox[3]) / 2,
            )
            question_cross_area_ratio = (
                _bbox_area(question_bbox) / cross_area if cross_area else 0.0
            )
            question_cross_width_ratio = (
                (question_bbox[2] - question_bbox[0]) / cross_width
                if cross_width
                else 0.0
            )
            question_cross_height_ratio = (
                (question_bbox[3] - question_bbox[1]) / cross_height
                if cross_height
                else 0.0
            )
            cross_center_gap_ratio = _point_bbox_gap_distance(
                cross_center, question_bbox
            )
            question_cross_gap_ratio = _bbox_gap_distance(
                question_bbox, cross_bbox
            )
        items.append(
            {
                "cross_id": cross_id,
                "source": anchor["source"],
                "independent_scan_supported": anchor.get(
                    "independent_scan_supported", False
                ),
                "matched": first_item.matched,
                "confidence": first_item.confidence,
                "expected_truth_id": expected_truth_id,
                "localized_truth_id": localized_truth_id,
                "localization_status": localization_status,
                "question_cross_area_ratio": (
                    round(question_cross_area_ratio, 6)
                    if question_cross_area_ratio is not None
                    else None
                ),
                "question_cross_width_ratio": (
                    round(question_cross_width_ratio, 6)
                    if question_cross_width_ratio is not None
                    else None
                ),
                "question_cross_height_ratio": (
                    round(question_cross_height_ratio, 6)
                    if question_cross_height_ratio is not None
                    else None
                ),
                "cross_center_gap_ratio": (
                    round(cross_center_gap_ratio, 6)
                    if cross_center_gap_ratio is not None
                    else None
                ),
                "question_cross_gap_ratio": (
                    round(question_cross_gap_ratio, 6)
                    if question_cross_gap_ratio is not None
                    else None
                ),
                "geometry_reasons": geometry_reasons_by_id.get(cross_id, []),
            }
        )
    true_anchor_items = [
        item for item in items if item["expected_truth_id"] is not None
    ]
    return {
        "true_anchor_count": len(true_anchor_items),
        "true_anchor_localization_failure_count": sum(
            item["localization_status"] != "correct_truth"
            for item in true_anchor_items
        ),
        "false_anchor_matched_count": sum(
            item["localization_status"] == "matched_false_anchor"
            for item in items
        ),
        "items": items,
        "policy": (
            "Truth-linked risk labels are diagnostic-only and never influence "
            "retry selection or event output."
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

    def recognize_holistic_v3_geometry(
        self,
        image_path: str,
        *,
        prompt_version: str,
    ) -> HolisticCrossGeometryResult:
        if prompt_version != "HOLISTIC_CROSS_GEOMETRY_PROMPT_V3":
            raise ValueError(f"unsupported holistic V3 prompt: {prompt_version}")
        diagnostic = {
            "operation": "holistic_v3_geometry",
            "prompt_version": prompt_version,
            "image_max_edge": self.client.max_edge,
            "max_retries": self.client.max_retries,
        }
        image_url = prepare_image_data_url(
            image_path,
            self.client.max_edge,
            self.client.jpeg_quality,
            diagnostic,
        )
        return self._call(
            "holistic_v3_geometry",
            image_path,
            {
                "prompt_version": prompt_version,
                "image_max_edge": self.client.max_edge,
                "max_retries": self.client.max_retries,
            },
            lambda: self.client._request(
                {
                    "prompt": HOLISTIC_CROSS_GEOMETRY_PROMPT_V3,
                    "image_url": image_url,
                },
                HolisticCrossGeometryResult,
                diagnostic,
            ),
        )

    def locate_spatial_cross_groups(
        self,
        image_path: str,
        anchors: list[dict],
        subject_hint: str | None,
    ) -> SpatialQuestionGroupResult:
        prompt = SPATIAL_CROSS_GROUP_PROMPT.replace(
            "__CROSSES__",
            json.dumps(anchors, ensure_ascii=False, indent=2),
        )
        diagnostic = {
            "operation": "spatial_cross_group_localization",
            "cross_count": len(anchors),
        }
        image_url = prepare_image_data_url(
            image_path,
            self.client.max_edge,
            self.client.jpeg_quality,
            diagnostic,
        )
        return self._call(
            "spatial_cross_group_localization",
            image_path,
            {
                "cross_ids": [anchor["cross_id"] for anchor in anchors],
                "subject_hint": subject_hint,
            },
            lambda: self.client._request(
                {"prompt": prompt, "image_url": image_url},
                SpatialQuestionGroupResult,
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
    holistic_v3_experiment: dict | None = None,
) -> dict:
    pipeline_ran = pipeline is not None
    stable_experiment_ran = stable_experiment is not None
    cross_anchor_experiment_ran = cross_anchor_experiment is not None
    holistic_v3_experiment_ran = holistic_v3_experiment is not None
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
        "cross_anchor_profile": (cross_anchor_experiment or {}).get("profile"),
        "cross_anchor_llm1_input_mode": (cross_anchor_experiment or {}).get(
            "llm1_input_mode"
        ),
        "cross_anchor_independent_rescue_count": (
            cross_anchor_experiment or {}
        ).get("llm1_independent_rescue_count"),
        "cross_anchor_llm2_batch_error_count": (
            cross_anchor_experiment or {}
        ).get("llm2_batch_error_count"),
        "cross_anchor_preservation_confirmed_count": (
            cross_anchor_experiment or {}
        ).get("anchor_preservation_confirmed_count"),
        "cross_anchor_preservation_needs_review_count": (
            cross_anchor_experiment or {}
        ).get("anchor_preservation_needs_review_count"),
        "cross_anchor_preservation_confirmed_truth_recall": (
            cross_anchor_experiment or {}
        ).get("anchor_preservation_confirmed_truth_recall"),
        "cross_anchor_preservation_union_truth_recall": (
            cross_anchor_experiment or {}
        ).get("anchor_preservation_union_truth_recall"),
        "cross_anchor_preservation_silently_dropped_count": (
            cross_anchor_experiment or {}
        ).get("anchor_preservation_silently_dropped_count"),
        "cross_anchor_baseline_group_request_count": (
            cross_anchor_experiment or {}
        ).get("baseline_group_request_count"),
        "cross_anchor_spatial_request_budget_eligible": (
            cross_anchor_experiment or {}
        ).get("spatial_request_budget_eligible"),
        "cross_anchor_spatial_request_budget": (
            cross_anchor_experiment or {}
        ).get("spatial_request_budget"),
        "cross_anchor_spatial_experiment_status": (
            cross_anchor_experiment or {}
        ).get("spatial_experiment_status"),
        "cross_anchor_spatial_group_request_count": (
            cross_anchor_experiment or {}
        ).get("spatial_group_request_count"),
        "cross_anchor_spatial_group_error_count": (
            cross_anchor_experiment or {}
        ).get("spatial_group_error_count"),
        "cross_anchor_spatial_question_event_count": (
            cross_anchor_experiment or {}
        ).get("spatial_question_event_count"),
        "cross_anchor_spatial_truth_recall": (
            cross_anchor_experiment or {}
        ).get("spatial_truth_recall"),
        "cross_anchor_spatial_false_event_count": (
            cross_anchor_experiment or {}
        ).get("spatial_false_event_count"),
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
        "cross_anchor_llm1_model_confirmed_truth_recall": (
            cross_anchor_experiment or {}
        ).get("llm1_model_confirmed_truth_recall"),
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
        "cross_anchor_llm2_retry_suppressed_count": (
            cross_anchor_experiment or {}
        ).get("llm2_retry_suppressed_count"),
        "cross_anchor_llm2_retry_accepted_count": (
            cross_anchor_experiment or {}
        ).get("llm2_retry_accepted_count"),
        "cross_anchor_llm2_retry_rejected_count": (
            cross_anchor_experiment or {}
        ).get("llm2_retry_rejected_count"),
        "cross_anchor_llm2_first_pass_true_anchor_localization_failure_count": (
            cross_anchor_experiment or {}
        ).get("llm2_first_pass_true_anchor_localization_failure_count"),
        "cross_anchor_llm2_first_pass_false_anchor_matched_count": (
            cross_anchor_experiment or {}
        ).get("llm2_first_pass_false_anchor_matched_count"),
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
        "holistic_v3_prompt_version": (holistic_v3_experiment or {}).get(
            "prompt_version"
        ),
        "holistic_v3_long_edge": (holistic_v3_experiment or {}).get("long_edge"),
        "holistic_v3_json_schema_status": (holistic_v3_experiment or {}).get(
            "json_schema_status"
        ),
        "holistic_v3_cross_anchor_count": (holistic_v3_experiment or {}).get(
            "cross_anchor_count"
        ),
        "holistic_v3_cross_anchor_truth_matched_count": (
            holistic_v3_experiment or {}
        ).get("cross_anchor_truth_matched_count"),
        "holistic_v3_cross_anchor_truth_recall": (
            holistic_v3_experiment or {}
        ).get("cross_anchor_truth_recall"),
        "holistic_v3_cross_anchor_false_count": (
            holistic_v3_experiment or {}
        ).get("cross_anchor_false_count"),
        "holistic_v3_question_event_count": (holistic_v3_experiment or {}).get(
            "question_event_count"
        ),
        "holistic_v3_question_event_truth_matched_count": (
            holistic_v3_experiment or {}
        ).get("question_event_truth_matched_count"),
        "holistic_v3_question_event_truth_recall": (
            holistic_v3_experiment or {}
        ).get("question_event_truth_recall"),
        "holistic_v3_question_event_false_count": (
            holistic_v3_experiment or {}
        ).get("question_event_false_count"),
        "holistic_v3_circle_only_false_positive_count": (
            holistic_v3_experiment or {}
        ).get("circle_only_false_positive_count"),
        "holistic_v3_duplicate_event_count": (holistic_v3_experiment or {}).get(
            "duplicate_event_count"
        ),
        "holistic_v3_unassigned_cross_anchor_count": (
            holistic_v3_experiment or {}
        ).get("unassigned_cross_anchor_count"),
        "holistic_v3_llm_request_count": (holistic_v3_experiment or {}).get(
            "llm_request_count"
        ),
        "holistic_v3_stage_timings_ms": (holistic_v3_experiment or {}).get(
            "timings_ms"
        ),
        "holistic_v3_content_ocr_status": (
            (holistic_v3_experiment or {}).get("content_ocr_status")
            if holistic_v3_experiment_ran
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
        "holistic_v3_experiment_status": (
            "completed" if holistic_v3_experiment_ran else "not_run"
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
    cross_anchor_confirmation_only: bool = False,
    holistic_v3_only: bool = False,
    holistic_v3_long_edge: int | None = None,
    cross_anchor_profile: str = "baseline",
    cross_anchor_replay_from: Path | None = None,
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
        if cross_cv_config is not None and not holistic_v3_only
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
    holistic_v3_experiment = None
    holistic_v3_experiment_error = None
    holistic_v3_recorder = None
    error = None
    if cross_anchor_confirmation_only:
        recorder = ExchangeRecorder(case_dir)
        client = MiniMaxVisionClient.from_settings()
        client.diagnostic_event_sink = recorder
        recording_client = RecordingVisionClient(client, recorder)
        phase_started = time.perf_counter()
        try:
            if cross_cv_config is None or truth_regions is None:
                raise ValueError(
                    "cross-anchor confirmation requires CV config and truth regions"
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
                profile=cross_anchor_profile,
                confirmation_only=True,
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
    if not cv_only and not holistic_v3_only and not cross_anchor_confirmation_only:
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
                    profile=cross_anchor_profile,
                    llm1_snapshot_dir=(
                        resolve_cross_anchor_snapshot_dir(
                            cross_anchor_replay_from,
                            label,
                        )
                        if cross_anchor_replay_from is not None
                        else None
                    ),
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
    if holistic_v3_only:
        phase_started = time.perf_counter()
        try:
            if (
                cross_cv_config is None
                or truth_regions is None
                or holistic_v3_long_edge is None
            ):
                raise ValueError(
                    "holistic V3 requires config, truth regions, and long edge"
                )
            holistic_v3_recorder = ExchangeRecorder(case_dir)
            client = MiniMaxVisionClient.from_settings()
            client.max_edge = holistic_v3_long_edge
            client.max_retries = int(cross_cv_config["holistic_v3_max_retries"])
            client.diagnostic_event_sink = holistic_v3_recorder
            recording_client = RecordingVisionClient(client, holistic_v3_recorder)
            _write_json(
                case_dir / "holistic-v3-effective-config.json",
                {
                    "prompt_version": cross_cv_config[
                        "holistic_v3_prompt_version"
                    ],
                    "long_edge": holistic_v3_long_edge,
                    "max_retries": cross_cv_config["holistic_v3_max_retries"],
                    "question_truth_min_iou": cross_cv_config[
                        "question_truth_min_iou"
                    ],
                    "truth_match_margin_ratio": cross_cv_config[
                        "truth_match_margin_ratio"
                    ],
                },
            )
            holistic_v3_experiment = run_holistic_v3_experiment(
                image_path=image_path,
                case_dir=case_dir,
                client=recording_client,
                truth_regions=truth_regions,
                config=cross_cv_config,
                long_edge=holistic_v3_long_edge,
            )
        except Exception as exc:
            holistic_v3_experiment_error = {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "message": str(exc),
                "diagnostic": getattr(exc, "diagnostic", None),
            }
            _write_json(
                case_dir / "holistic-v3-experiment-error.json",
                holistic_v3_experiment_error,
            )
        finally:
            timings_ms["holistic_v3_experiment"] = round(
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
        holistic_v3_experiment=holistic_v3_experiment,
    )
    summary["error"] = error
    summary["stable_experiment_error"] = stable_experiment_error
    summary["cross_anchor_experiment_error"] = cross_anchor_experiment_error
    summary["holistic_v3_experiment_error"] = holistic_v3_experiment_error
    if holistic_v3_only:
        summary["holistic_v3_experiment_status"] = (
            "failed" if holistic_v3_experiment_error is not None else "completed"
        )
        summary["checkpoints"]["holistic_v3_prompt_version"] = cross_cv_config[
            "holistic_v3_prompt_version"
        ]
        summary["checkpoints"]["holistic_v3_long_edge"] = holistic_v3_long_edge
        summary["checkpoints"]["holistic_v3_json_schema_status"] = (
            "failed" if holistic_v3_experiment_error is not None else "completed"
        )
        summary["checkpoints"]["holistic_v3_llm_request_count"] = (
            holistic_v3_recorder.call_count
            if holistic_v3_recorder is not None
            else 0
        )
        summary["checkpoints"]["holistic_v3_content_ocr_status"] = "not_run"
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
    profile: str = "baseline",
    llm1_snapshot_dir: Path | None = None,
    confirmation_only: bool = False,
) -> dict:
    if profile not in CROSS_ANCHOR_PROFILES:
        raise ValueError(f"unknown cross-anchor profile: {profile}")
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
    llm1_snapshot = (
        load_cross_anchor_llm1_snapshot(llm1_snapshot_dir)
        if llm1_snapshot_dir is not None
        else None
    )
    verification_runs = []
    phase_started = time.perf_counter()
    if llm1_snapshot is not None:
        verification_run_count = 0
        verification_runs.append(llm1_snapshot["verification"])
        _write_json(
            experiment_dir / "llm1-replay-source.json",
            {
                "source": llm1_snapshot["source"],
                "sha256": llm1_snapshot["sha256"],
            },
        )
    else:
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
    independent_scan = (
        llm1_snapshot["independent_scan"]
        if llm1_snapshot is not None
        else client.scan_independent_crosses(str(image_path))
    )
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
    if llm1_snapshot is not None:
        fallback_verification = llm1_snapshot["fallback_verification"]
    elif fallback_candidates:
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
    retained_independent_scan = IndependentCrossScanResult(
        crosses=[
            cross
            for candidate_id, cross in enumerate(independent_scan.crosses)
            if candidate_id in retained_fallback_ids
        ]
    )
    rescue_audit = None
    if profile == "baseline":
        verified_independent_scan = IndependentCrossScanResult(
            crosses=(
                retained_independent_scan.crosses
                if fallback_generates_anchors
                else []
            )
        )
    else:
        preliminary_anchors = build_cross_anchors(
            verification,
            IndependentCrossScanResult(crosses=[]),
            cv_candidates,
            config,
        )
        verified_independent_scan, rescue_audit = select_independent_rescue_crosses(
            existing_anchors=preliminary_anchors,
            independent_scan=independent_scan,
            fallback_verification=fallback_verification,
            image_path=image_path,
            config=config,
        )
        _write_json(
            experiment_dir / "independent-rescue-audit.json",
            rescue_audit,
        )
    anchors = build_cross_anchors(
        verification,
        verified_independent_scan,
        cv_candidates,
        config,
    )
    if profile != "baseline":
        for anchor in anchors:
            if anchor["source"] == "llm_fallback":
                anchor["source"] = "independent_scan_rescue"
    mark_existing_anchor_scan_support(
        anchors,
        retained_independent_scan if profile == "baseline" else independent_scan,
        min_iou=float(config["cross_anchor_fallback_merge_iou_threshold"]),
        max_center_distance_ratio=float(
            config["cross_anchor_fallback_merge_center_distance_ratio"]
        ),
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
    llm1_model_confirmed_truth_comparison = compare_cross_candidates_to_truth(
        [
            candidate
            for candidate in cv_candidates
            if any(
                verdict.candidate_id == candidate["candidate_id"]
                and verdict.disposition == "confirmed"
                for verdict in verification.verdicts
            )
        ],
        truth_regions,
        margin_ratio=float(config.get("truth_match_margin_ratio", 0.0)),
    )
    _write_json(
        experiment_dir / "llm1-model-confirmed-truth-comparison.json",
        llm1_model_confirmed_truth_comparison,
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

    if confirmation_only:
        timings_ms["total"] = round(
            (time.perf_counter() - experiment_started) * 1000, 2
        )
        disposition_counts = {
            disposition: sum(
                verdict.disposition == disposition
                for verdict in verification.verdicts
            )
            for disposition in ("confirmed", "rejected", "uncertain")
        }
        summary = {
            "mode": "confirmation_only",
            "profile": profile,
            "llm1_input_mode": "replay" if llm1_snapshot is not None else "live",
            "llm1_snapshot_source": (
                llm1_snapshot["source"] if llm1_snapshot is not None else None
            ),
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
            "llm1_independent_rescue_count": (
                rescue_audit["rescued_count"] if rescue_audit is not None else 0
            ),
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
            "llm1_truth_matched_count": llm1_truth_comparison[
                "matched_truth_count"
            ],
            "llm1_truth_recall": llm1_truth_comparison["truth_recall"],
            "llm1_model_confirmed_truth_matched_count": (
                llm1_model_confirmed_truth_comparison["matched_truth_count"]
            ),
            "llm1_model_confirmed_truth_recall": (
                llm1_model_confirmed_truth_comparison["truth_recall"]
            ),
            "llm1_false_cross_count": len(
                llm1_truth_comparison["false_candidate_ids"]
            ),
            "llm1_duplicate_truth_candidate_count": len(
                llm1_truth_multiplicity
            ),
            "llm_request_count": (
                0
                if llm1_snapshot is not None
                else verification_run_count
                + 1
                + (1 if fallback_candidates else 0)
            ),
            "timings_ms": timings_ms,
            "content_ocr_status": "not_run",
        }
        _write_json(experiment_dir / "timings.json", timings_ms)
        _write_json(experiment_dir / "summary.json", summary)
        return summary

    batch_size = int(config["cross_anchor_llm2_batch_size"])
    llm2_run_count = int(config.get("cross_anchor_llm2_localization_runs", 1))
    llm2_batch_count = 0
    llm2_started = time.perf_counter()
    llm2_runs = []
    assignment_audits = []
    geometry_audits = []
    run_started = time.perf_counter()
    anchored_items = []
    llm2_batch_errors = []
    spatial_localization = None
    spatial_request_budget = None
    spatial_experiment_status = None
    baseline_group_request_count = math.ceil(len(anchors) / batch_size) if anchors else 0
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
        try:
            batch_result = client.locate_cross_anchored_questions(
                str(batch_overlay_path),
                batch,
                subject_hint,
            )
        except Exception as exc:
            if profile not in ("anchor-preserving", "spatial-grouped"):
                raise
            llm2_batch_errors.append(
                {
                    "batch_index": batch_index,
                    "cross_ids": [anchor["cross_id"] for anchor in batch],
                    "type": type(exc).__name__,
                    "code": getattr(exc, "code", None),
                    "message": str(exc),
                }
            )
            batch_result = CrossAnchoredQuestionResult(
                items=[
                    CrossAnchoredQuestion(
                        cross_id=anchor["cross_id"],
                        matched=False,
                        question_bbox=None,
                        unmatched_reason="LLM2 batch request failed",
                        confidence=0.0,
                    )
                    for anchor in batch
                ]
            )
        anchored_items.extend(batch_result.items)
    _write_json(
        experiment_dir / "llm2-batch-errors.json",
        {"errors": llm2_batch_errors},
    )
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
    timings_ms["baseline_llm2_localization"] = round(
        (time.perf_counter() - run_started) * 1000,
        2,
    )
    anchor_preservation = None
    anchor_preservation_truth = None
    anchor_preservation_confirmed_truth = None
    spatial_deduplicated_events = None
    spatial_truth_comparison = None
    if profile in ("anchor-preserving", "spatial-grouped"):
        anchor_preservation = build_anchor_preservation_events(
            anchors=anchors,
            result=first_run,
            geometry_audit=first_geometry_audit,
            batch_errors=llm2_batch_errors,
            image_path=image_path,
            config=config,
        )
        preservation_events = {
            "event_count": anchor_preservation["event_count"],
            "events": anchor_preservation["events"],
            "policy": anchor_preservation["policy"],
        }
        _write_json(
            experiment_dir / "anchor-preservation-events.json",
            preservation_events,
        )
        _write_json(
            experiment_dir / "anchor-preservation-audit.json",
            {
                "items": anchor_preservation["audit"],
                "batch_errors": llm2_batch_errors,
            },
        )
        anchor_preservation_truth = compare_question_events_to_truth(
            preservation_events,
            truth_regions,
            min_iou=float(config["question_truth_min_iou"]),
        )
        _write_json(
            experiment_dir / "anchor-preservation-truth-comparison.json",
            anchor_preservation_truth,
        )
        confirmed_events = [
            event
            for event in anchor_preservation["events"]
            if event["status"] == "confirmed"
        ]
        confirmed_event_result = {
            "event_count": len(confirmed_events),
            "events": confirmed_events,
        }
        anchor_preservation_confirmed_truth = compare_question_events_to_truth(
            confirmed_event_result,
            truth_regions,
            min_iou=float(config["question_truth_min_iou"]),
        )
        _write_json(
            experiment_dir
            / "anchor-preservation-confirmed-truth-comparison.json",
            anchor_preservation_confirmed_truth,
        )
    if profile == "spatial-grouped":
        spatial_groups = group_cross_anchors_spatially(anchors, config)
        _write_json(
            experiment_dir / "spatial-anchor-groups.json",
            spatial_groups,
        )
        spatial_request_budget = audit_spatial_request_budget(
            anchor_count=len(anchors),
            baseline_batch_size=batch_size,
            spatial_group_count=len(spatial_groups),
        )
        _write_json(
            experiment_dir / "spatial-request-budget-audit.json",
            spatial_request_budget,
        )
        if spatial_request_budget["eligible"]:
            spatial_started = time.perf_counter()
            spatial_localization = run_spatial_group_localization(
                image_path=image_path,
                experiment_dir=experiment_dir,
                client=client,
                groups=spatial_groups,
                subject_hint=subject_hint,
            )
            timings_ms["spatial_llm2_localization"] = round(
                (time.perf_counter() - spatial_started) * 1000,
                2,
            )
            llm2_batch_count += spatial_localization["request_count"]
            spatial_result = spatial_localization["result"]
            spatial_experiment_status = "completed"
            _write_json(
                experiment_dir / "spatial-group-result.json",
                spatial_result,
            )
            _write_json(
                experiment_dir / "group-membership-audit.json",
                spatial_localization["membership_audit"],
            )
            _write_json(
                experiment_dir / "spatial-group-errors.json",
                {"errors": spatial_localization["errors"]},
            )
            spatial_anchored_result = CrossAnchoredQuestionResult(
                items=[
                    CrossAnchoredQuestion(
                        cross_id=cross_id,
                        matched=True,
                        question_bbox=question_group.question_bbox,
                        unmatched_reason=None,
                        confidence=question_group.confidence,
                    )
                    for question_group in spatial_result.groups
                    for cross_id in question_group.cross_ids
                ]
                + [
                    CrossAnchoredQuestion(
                        cross_id=item.cross_id,
                        matched=False,
                        question_bbox=None,
                        unmatched_reason=item.reason,
                        confidence=item.confidence,
                    )
                    for item in spatial_result.unmatched
                ]
            )
            spatial_geometry_audit = audit_anchored_question_geometry(
                spatial_anchored_result,
                anchors,
                max_area_ratio=float(
                    config["cross_anchor_question_max_area_ratio"]
                ),
                max_gap_ratio=float(config["cross_anchor_question_max_gap_ratio"]),
            )
            spatial_batch_errors = [
                {
                    "batch_index": error["spatial_group_id"] + 1,
                    "cross_ids": error["cross_ids"],
                    "type": error["type"],
                    "code": error["code"],
                    "message": error["message"],
                }
                for error in spatial_localization["errors"]
            ]
            spatial_anchor_preservation = build_anchor_preservation_events(
                anchors=anchors,
                result=spatial_anchored_result,
                geometry_audit=spatial_geometry_audit,
                batch_errors=spatial_batch_errors,
                image_path=image_path,
                config=config,
            )
            _write_json(
                experiment_dir / "spatial-anchor-preservation-events.json",
                {
                    "event_count": spatial_anchor_preservation["event_count"],
                    "events": spatial_anchor_preservation["events"],
                    "policy": spatial_anchor_preservation["policy"],
                },
            )
            _write_json(
                experiment_dir / "spatial-anchor-preservation-audit.json",
                {
                    "items": spatial_anchor_preservation["audit"],
                    "silently_dropped_anchor_count": spatial_anchor_preservation[
                        "silently_dropped_anchor_count"
                    ],
                    "batch_errors": spatial_batch_errors,
                },
            )
            spatial_question_events = build_spatial_question_events(
                spatial_result,
                spatial_anchor_preservation,
            )
            _write_json(
                experiment_dir / "spatial-question-events.json",
                spatial_question_events,
            )
            spatial_deduplicated_events = deduplicate_spatial_question_events(
                spatial_question_events["events"],
                anchors,
                config,
            )
            _write_json(
                experiment_dir / "deduplicated-question-events.json",
                {
                    "event_count": spatial_deduplicated_events["event_count"],
                    "events": spatial_deduplicated_events["events"],
                    "policy": spatial_deduplicated_events["policy"],
                },
            )
            _write_json(
                experiment_dir / "deduplication-audit.json",
                {"decisions": spatial_deduplicated_events["audit"]},
            )
            spatial_truth_comparison = compare_question_events_to_truth(
                spatial_deduplicated_events,
                truth_regions,
                min_iou=float(config["question_truth_min_iou"]),
            )
            _write_json(
                experiment_dir / "spatial-question-events-truth-comparison.json",
                spatial_truth_comparison,
            )
        else:
            spatial_experiment_status = "budget_rejected"
            timings_ms["spatial_llm2_localization"] = 0.0
    timings_ms["llm2_localization_run_001"] = timings_ms[
        "baseline_llm2_localization"
    ]

    retry_selection = select_llm2_retry_anchors(
        first_run,
        anchors,
        first_geometry_audit,
        min_center_gap_ratio=float(
            config["cross_anchor_llm2_retry_min_center_gap_ratio"]
        ),
        min_first_pass_question_cross_area_ratio=float(
            config[
                "cross_anchor_llm2_retry_first_pass_min_question_cross_area_ratio"
            ]
        ),
        max_question_cross_iou=float(
            config["cross_anchor_llm2_retry_max_question_cross_iou"]
        ),
        shared_question_iou_threshold=float(
            config[
                "cross_anchor_llm2_retry_shared_question_iou_threshold"
            ]
        ),
        shared_question_min_anchor_distance_ratio=float(
            config[
                "cross_anchor_llm2_retry_shared_question_min_anchor_distance_ratio"
            ]
        ),
        shared_question_min_group_size=int(
            config[
                "cross_anchor_llm2_retry_shared_question_min_group_size"
            ]
        ),
        max_retry_requests=int(
            config["cross_anchor_llm2_retry_max_requests_per_page"]
        ),
    )
    _write_json(experiment_dir / "llm2-retry-selection.json", retry_selection)
    retry_request_count = 0
    retry_decision_audit = {
        "accepted_count": 0,
        "rejected_count": 0,
        "min_question_cross_area_ratio": float(
            config["cross_anchor_llm2_retry_min_question_cross_area_ratio"]
        ),
        "max_question_cross_iou": float(
            config["cross_anchor_llm2_retry_max_question_cross_iou"]
        ),
        "duplicate_question_iou_threshold": float(
            config["cross_anchor_duplicate_question_iou_threshold"]
        ),
        "decisions": [],
        "policy": (
            "Retry output is a challenger: only expanded, non-copying, novel "
            "question regions may join first-pass events."
        ),
    }
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
        accepted_retry_run, retry_decision_audit = decide_llm2_retry_results(
            first_run,
            retry_run,
            retry_selection["anchors"],
            min_question_cross_area_ratio=float(
                config[
                    "cross_anchor_llm2_retry_min_question_cross_area_ratio"
                ]
            ),
            max_question_cross_iou=float(
                config["cross_anchor_llm2_retry_max_question_cross_iou"]
            ),
            duplicate_question_iou_threshold=float(
                config["cross_anchor_duplicate_question_iou_threshold"]
            ),
        )
        llm2_runs.append(accepted_retry_run)
        _write_json(
            experiment_dir / "llm2-accepted-retry-questions.json",
            accepted_retry_run,
        )
        timings_ms["llm2_localization_run_002"] = round(
            (time.perf_counter() - retry_started) * 1000,
            2,
        )
    _write_json(
        experiment_dir / "llm2-retry-decision-audit.json",
        retry_decision_audit,
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
    first_pass_risk_audit = audit_first_pass_llm2_localization_risk(
        first_run,
        anchors,
        llm1_truth_comparison,
        truth_comparison,
        first_geometry_audit,
    )
    _write_json(
        experiment_dir / "llm2-first-pass-risk-audit.json",
        first_pass_risk_audit,
    )
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
        "profile": profile,
        "llm1_input_mode": "replay" if llm1_snapshot is not None else "live",
        "llm1_snapshot_source": (
            llm1_snapshot["source"] if llm1_snapshot is not None else None
        ),
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
        "llm1_independent_rescue_count": (
            rescue_audit["rescued_count"] if rescue_audit is not None else 0
        ),
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
        "llm1_model_confirmed_truth_matched_count": (
            llm1_model_confirmed_truth_comparison["matched_truth_count"]
        ),
        "llm1_model_confirmed_truth_recall": (
            llm1_model_confirmed_truth_comparison["truth_recall"]
        ),
        "llm1_false_cross_count": len(
            llm1_truth_comparison["false_candidate_ids"]
        ),
        "llm1_duplicate_truth_candidate_count": len(
            llm1_truth_multiplicity
        ),
        "llm2_matched_question_count": len(matched_questions),
        "llm2_unmatched_cross_count": len(anchored_questions.items)
        - len(matched_questions),
        "llm2_batch_error_count": len(llm2_batch_errors),
        "anchor_preservation_confirmed_count": (
            sum(
                event["status"] == "confirmed"
                for event in anchor_preservation["events"]
            )
            if anchor_preservation is not None
            else None
        ),
        "anchor_preservation_needs_review_count": (
            sum(
                event["status"] == "needs_review"
                for event in anchor_preservation["events"]
            )
            if anchor_preservation is not None
            else None
        ),
        "anchor_preservation_confirmed_truth_recall": (
            anchor_preservation_confirmed_truth["truth_recall"]
            if anchor_preservation_confirmed_truth is not None
            else None
        ),
        "anchor_preservation_union_truth_recall": (
            anchor_preservation_truth["truth_recall"]
            if anchor_preservation_truth is not None
            else None
        ),
        "anchor_preservation_silently_dropped_count": (
            anchor_preservation["silently_dropped_anchor_count"]
            if anchor_preservation is not None
            else None
        ),
        "baseline_group_request_count": baseline_group_request_count,
        "spatial_request_budget_eligible": (
            spatial_request_budget["eligible"]
            if spatial_request_budget is not None
            else None
        ),
        "spatial_request_budget": (
            spatial_request_budget["baseline_request_budget"]
            if spatial_request_budget is not None
            else None
        ),
        "spatial_experiment_status": spatial_experiment_status,
        "spatial_group_request_count": (
            spatial_localization["request_count"]
            if spatial_localization is not None
            else (0 if profile == "spatial-grouped" else None)
        ),
        "spatial_group_error_count": (
            len(spatial_localization["errors"])
            if spatial_localization is not None
            else (0 if profile == "spatial-grouped" else None)
        ),
        "spatial_question_event_count": (
            spatial_deduplicated_events["event_count"]
            if spatial_deduplicated_events is not None
            else None
        ),
        "spatial_truth_recall": (
            spatial_truth_comparison["truth_recall"]
            if spatial_truth_comparison is not None
            else None
        ),
        "spatial_false_event_count": (
            len(spatial_truth_comparison["false_event_ids"])
            if spatial_truth_comparison is not None
            else None
        ),
        "llm2_assignment_audit_valid": assignment_audit["valid"],
        "llm2_localization_run_count": len(llm2_runs),
        "llm2_retry_trigger_count": retry_selection["trigger_count"],
        "llm2_retry_request_count": retry_request_count,
        "llm2_retry_suppressed_count": len(retry_selection["suppressed"]),
        "llm2_retry_accepted_count": retry_decision_audit["accepted_count"],
        "llm2_retry_rejected_count": retry_decision_audit["rejected_count"],
        "llm2_first_pass_true_anchor_localization_failure_count": (
            first_pass_risk_audit[
                "true_anchor_localization_failure_count"
            ]
        ),
        "llm2_first_pass_false_anchor_matched_count": first_pass_risk_audit[
            "false_anchor_matched_count"
        ],
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
            (
                0
                if llm1_snapshot is not None
                else verification_run_count
                + 1
                + (1 if fallback_candidates else 0)
            )
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
        "cross_anchor_llm2_retry_first_pass_min_question_cross_area_ratio",
        "cross_anchor_llm2_retry_min_question_cross_area_ratio",
        "cross_anchor_llm2_retry_max_question_cross_iou",
        "cross_anchor_llm2_retry_shared_question_iou_threshold",
        "cross_anchor_llm2_retry_shared_question_min_anchor_distance_ratio",
        "cross_anchor_llm2_retry_shared_question_min_group_size",
        "cross_anchor_llm2_retry_max_requests_per_page",
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
        "cross_anchor_rescue_min_scan_confidence",
        "cross_anchor_rescue_min_red_pixel_ratio",
        "cross_anchor_rescue_min_bbox_area_ratio",
        "cross_anchor_rescue_max_bbox_area_ratio",
        "cross_anchor_rescue_edge_margin_ratio",
        "cross_anchor_preserve_strong_sources",
        "cross_anchor_preserve_min_confidence",
        "cross_anchor_preserve_min_red_pixel_ratio",
        "cross_anchor_context_horizontal_padding_ratio",
        "cross_anchor_context_vertical_padding_ratio",
        "cross_anchor_context_min_width_ratio",
        "cross_anchor_context_min_height_ratio",
        "cross_anchor_context_max_area_ratio",
        "cross_anchor_spatial_row_distance_ratio",
        "cross_anchor_spatial_horizontal_gap_ratio",
        "cross_anchor_spatial_max_anchors_per_group",
        "cross_anchor_spatial_crop_padding_ratio",
        "cross_anchor_spatial_max_crop_area_ratio",
        "cross_anchor_spatial_dedupe_iou_threshold",
        "cross_anchor_spatial_dedupe_containment_threshold",
        "cross_anchor_spatial_dedupe_max_anchor_distance_ratio",
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
        "cross_anchor_llm2_retry_shared_question_min_group_size",
        "cross_anchor_llm2_retry_max_requests_per_page",
        "cross_anchor_spatial_max_anchors_per_group",
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
        "cross_anchor_llm2_retry_first_pass_min_question_cross_area_ratio",
        "cross_anchor_llm2_retry_min_question_cross_area_ratio",
        *boolean_fields,
        "cross_anchor_preserve_strong_sources",
    }
    for name in ratio_fields:
        if not 0 <= float(config[name]) <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    for name in (
        "cross_anchor_llm2_retry_first_pass_min_question_cross_area_ratio",
        "cross_anchor_llm2_retry_min_question_cross_area_ratio",
    ):
        if float(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(config["arm_inner_radius_ratio"]) >= float(
        config["arm_outer_radius_ratio"]
    ):
        raise ValueError("arm inner radius must be smaller than outer radius")
    strong_sources = config["cross_anchor_preserve_strong_sources"]
    if not isinstance(strong_sources, list) or not strong_sources:
        raise ValueError("cross anchor preserve strong sources must not be empty")
    allowed_sources = {
        "cv_confirmed",
        "cv_uncertain",
        "cv_high_score_retained",
        "cv_rejected_retained",
        "independent_scan_rescue",
    }
    if any(
        not isinstance(source, str) or source not in allowed_sources
        for source in strong_sources
    ):
        raise ValueError("cross anchor preserve strong sources contain invalid value")
    if float(config["cross_anchor_rescue_min_bbox_area_ratio"]) > float(
        config["cross_anchor_rescue_max_bbox_area_ratio"]
    ):
        raise ValueError("cross anchor rescue minimum area exceeds maximum")

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


def run_holistic_v3_experiment(
    *,
    image_path: Path,
    case_dir: Path,
    client,
    truth_regions: list[dict],
    config: dict,
    long_edge: int,
) -> dict:
    experiment_started = time.perf_counter()
    experiment_dir = case_dir / "holistic-v3-experiment"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    prompt_version = str(config["holistic_v3_prompt_version"])

    request_started = time.perf_counter()
    result = client.recognize_holistic_v3_geometry(
        str(image_path),
        prompt_version=prompt_version,
    )
    request_ms = round((time.perf_counter() - request_started) * 1000, 2)
    _write_json(experiment_dir / "result.json", result)

    anchor_candidates = [
        {
            "candidate_id": anchor.anchor_id,
            "bbox": list(anchor.bbox),
            "center": [
                round((anchor.bbox[0] + anchor.bbox[2]) / 2, 6),
                round((anchor.bbox[1] + anchor.bbox[3]) / 2, 6),
            ],
        }
        for anchor in result.cross_anchors
    ]
    anchor_comparison = compare_cross_candidates_to_truth(
        anchor_candidates,
        truth_regions,
        margin_ratio=float(config["truth_match_margin_ratio"]),
    )
    _write_json(
        experiment_dir / "cross-anchor-truth-comparison.json",
        anchor_comparison,
    )

    event_audit = {
        "events": [
            {
                "event_id": index,
                "source_event_id": event.event_id,
                "anchor_ids": list(event.anchor_ids),
                "question_bboxes": [list(event.question_bbox)],
            }
            for index, event in enumerate(result.question_events)
        ]
    }
    event_comparison = compare_question_events_to_truth(
        event_audit,
        truth_regions,
        min_iou=float(config["question_truth_min_iou"]),
    )
    _write_json(
        experiment_dir / "question-event-truth-comparison.json",
        event_comparison,
    )

    circle_candidates = [
        {
            "candidate_id": index,
            "bbox": list(region.bbox),
            "center": [
                round((region.bbox[0] + region.bbox[2]) / 2, 6),
                round((region.bbox[1] + region.bbox[3]) / 2, 6),
            ],
        }
        for index, region in enumerate(result.circle_only_regions)
    ]
    circle_comparison = compare_cross_candidates_to_truth(
        circle_candidates,
        truth_regions,
        margin_ratio=float(config["truth_match_margin_ratio"]),
    )
    _write_json(
        experiment_dir / "circle-only-truth-comparison.json",
        circle_comparison,
    )

    anchor_entries = [
        {"mark_id": anchor.anchor_id, "bbox": anchor.bbox}
        for anchor in result.cross_anchors
    ]
    event_entries = [
        {"mark_id": event.event_id, "bbox": event.question_bbox}
        for event in result.question_events
    ]
    _draw_boxes(
        image_path,
        experiment_dir / "question-events-overlay.jpg",
        [("A", anchor_entries, "red"), ("Q", event_entries, "blue")],
    )

    timings_ms = {
        "model_request": request_ms,
        "total": round((time.perf_counter() - experiment_started) * 1000, 2),
    }
    summary = {
        "prompt_version": prompt_version,
        "long_edge": long_edge,
        "json_schema_status": "completed",
        "cross_anchor_count": len(result.cross_anchors),
        "cross_anchor_truth_matched_count": anchor_comparison["matched_truth_count"],
        "cross_anchor_truth_recall": anchor_comparison["truth_recall"],
        "cross_anchor_false_count": len(anchor_comparison["false_candidate_ids"]),
        "question_event_count": len(result.question_events),
        "question_event_truth_matched_count": event_comparison["matched_truth_count"],
        "question_event_truth_recall": event_comparison["truth_recall"],
        "question_event_false_count": len(event_comparison["false_event_ids"]),
        "duplicate_event_count": len(
            event_comparison["duplicate_truth_event_ids"]
        ),
        "circle_only_region_count": len(result.circle_only_regions),
        "circle_only_false_positive_count": circle_comparison[
            "matched_truth_count"
        ],
        "unassigned_cross_anchor_count": len(
            result.unassigned_cross_anchor_ids
        ),
        "llm_request_count": 1,
        "timings_ms": timings_ms,
        "content_ocr_status": "not_run",
    }
    _write_json(experiment_dir / "timings.json", timings_ms)
    _write_json(experiment_dir / "summary.json", summary)
    return summary


def load_holistic_v3_inputs(
    config_path: Path,
    truth_path: Path,
    labels: list[str],
) -> tuple[dict, dict[str, list[dict]]]:
    config, truth_by_label = load_cross_cv_inputs(config_path, truth_path, labels)
    required = {
        "holistic_v3_prompt_version",
        "holistic_v3_default_long_edge",
        "holistic_v3_supported_long_edges",
        "holistic_v3_max_retries",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"holistic V3 config missing fields: {missing}")
    if not isinstance(config["holistic_v3_prompt_version"], str) or not config[
        "holistic_v3_prompt_version"
    ]:
        raise ValueError("holistic V3 prompt version must not be empty")
    supported = config["holistic_v3_supported_long_edges"]
    if (
        not isinstance(supported, list)
        or not supported
        or any(not isinstance(value, int) or value <= 0 for value in supported)
        or len(supported) != len(set(supported))
    ):
        raise ValueError("holistic V3 supported long edges must be unique positives")
    default_long_edge = config["holistic_v3_default_long_edge"]
    if not isinstance(default_long_edge, int) or default_long_edge not in supported:
        raise ValueError("holistic V3 default long edge must be supported")
    max_retries = config["holistic_v3_max_retries"]
    if not isinstance(max_retries, int) or max_retries != 0:
        raise ValueError("holistic V3 max retries must be zero for Phase 0")
    return config, truth_by_label


def _write_report(output_dir: Path, summaries: list[dict]) -> None:
    lines = [
        "# 视觉识别流程诊断报告",
        "",
        "> CV 组件/证据组数量不等于错题数量；本表只用于定位首次数量偏差。",
        "",
        "| 图片 | 人工错题 | CV组件 | CV证据组 | 当前primitive | 当前事件 | 当前定位 | 当前内容 | 当前真值命中 | 当前真值召回 | 当前首次偏差 | 实验primitive | 稳定事件 | 重复event候选 | 重复primitive候选 | 跨单元圈叉候选 | 圈叉归属异常 | 未分配primitive | 未覆盖CV组件 | LLM实验耗时(ms) | 新方案CV候选 | 新方案确认红叉 | 保留uncertain | 保留高分CV | LLM漏检补充 | 复核通过fallback | 独立扫描红叉 | 独立扫描支持 | 本地几何合并 | LLM1确认真值召回 | 保留锚点真值召回 | LLM1区域外红叉 | LLM1真值重复 | 新方案错题定位 | 几何异常 | 重复错题候选 | 真值重复归属 | 新方案真值命中 | 新方案真值召回 | 内容/OCR状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            "| {label} | {expected} | {components} | {groups} | {current_primitives} | {marks} | {located} | {content} | {pipeline_truth_matched} | {pipeline_truth_recall} | {divergence} | {primitives} | {stable} | {duplicate_events} | {duplicate_primitives} | {geometry_candidates} | {membership_violations} | {unassigned} | {uncovered} | {experiment_ms} | {cross_candidates} | {confirmed_crosses} | {uncertain_retained} | {high_score_retained} | {fallback_crosses} | {fallback_verified} | {independent_scan} | {independent_supported} | {local_merges} | {llm1_confirmed_truth_recall} | {llm1_truth_recall} | {llm1_false_crosses} | {llm1_duplicate_truth} | {anchored_questions} | {anchored_geometry} | {duplicate_questions} | {duplicate_truth} | {truth_matched} | {truth_recall} | {content_ocr_status} |".format(
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
                llm1_confirmed_truth_recall=item[
                    "cross_anchor_llm1_model_confirmed_truth_recall"
                ],
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
            "> 首轮LLM2结果始终保留，只有经过本地审核合格的定向复查结果才会加入稳定事件；目标是稳定真值召回为1.0，允许存在误报事件。",
            "",
            "| 图片 | LLM1核验次数 | 不稳定CV候选 | 保留LLM1拒绝CV | fallback uncertain审计 | fallback生成锚点 | LLM2定位次数 | LLM2真锚定位失败 | 定向复查触发 | 定向复查请求 | 复查触发抑制 | 复查结果接纳 | 复查结果拒绝 | 第一次真值召回 | 第一次最小真值覆盖 | 定向复查新增找回 | 定向复查新增误报 | 稳定错题事件 | 稳定真值命中 | 稳定真值召回 | 稳定最小真值覆盖 | 稳定误报事件 | 新方案LLM请求 |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        item = summary["checkpoints"]
        lines.append(
            "| {label} | {runs} | {unstable} | {rejected} | {fallback_uncertain} | {fallback_generates} | {llm2_runs} | {llm2_true_failures} | {retry_triggers} | {retry_requests} | {retry_suppressed} | {retry_accepted} | {retry_rejected} | {first_recall} | {first_coverage} | {recovered} | {additional_false} | {events} | {matched} | {recall} | {union_coverage} | {false_events} | {requests} |".format(
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
                llm2_true_failures=item[
                    "cross_anchor_llm2_first_pass_true_anchor_localization_failure_count"
                ],
                retry_triggers=item["cross_anchor_llm2_retry_trigger_count"],
                retry_requests=item["cross_anchor_llm2_retry_request_count"],
                retry_suppressed=item[
                    "cross_anchor_llm2_retry_suppressed_count"
                ],
                retry_accepted=item[
                    "cross_anchor_llm2_retry_accepted_count"
                ],
                retry_rejected=item[
                    "cross_anchor_llm2_retry_rejected_count"
                ],
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
    lines.extend(
        [
            "",
            "## 独立实验 Profile",
            "",
            "> 同一输入回放时，LLM1输入显示 replay；空间实验先运行同锚点baseline，再仅在请求数不超过baseline预算时运行。",
            "",
            "| 图片 | Profile | LLM1输入 | 独立补锚 | baseline请求 | 空间预算 | 空间状态 | 空间请求 | 空间异常 | LLM2批次异常 | confirmed | needs_review | 静默丢锚 | confirmed召回 | 含保底召回 | 去重后事件 | 空间结果召回 | 空间误报事件 |",
            "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        item = summary["checkpoints"]
        lines.append(
            "| {label} | {profile} | {input_mode} | {rescue} | {baseline_requests} | {budget_eligible} | {spatial_status} | {spatial_requests} | {spatial_errors} | {batch_errors} | {confirmed} | {needs_review} | {dropped} | {confirmed_recall} | {union_recall} | {spatial_events} | {spatial_recall} | {spatial_false} |".format(
                label=summary["label"],
                profile=item["cross_anchor_profile"],
                input_mode=item["cross_anchor_llm1_input_mode"],
                rescue=item["cross_anchor_independent_rescue_count"],
                baseline_requests=item[
                    "cross_anchor_baseline_group_request_count"
                ],
                budget_eligible=item[
                    "cross_anchor_spatial_request_budget_eligible"
                ],
                spatial_status=item[
                    "cross_anchor_spatial_experiment_status"
                ],
                spatial_errors=item["cross_anchor_spatial_group_error_count"],
                batch_errors=item["cross_anchor_llm2_batch_error_count"],
                confirmed=item["cross_anchor_preservation_confirmed_count"],
                needs_review=item[
                    "cross_anchor_preservation_needs_review_count"
                ],
                dropped=item[
                    "cross_anchor_preservation_silently_dropped_count"
                ],
                confirmed_recall=item[
                    "cross_anchor_preservation_confirmed_truth_recall"
                ],
                union_recall=item[
                    "cross_anchor_preservation_union_truth_recall"
                ],
                spatial_requests=item[
                    "cross_anchor_spatial_group_request_count"
                ],
                spatial_events=item[
                    "cross_anchor_spatial_question_event_count"
                ],
                spatial_recall=item["cross_anchor_spatial_truth_recall"],
                spatial_false=item["cross_anchor_spatial_false_event_count"],
            )
        )
    lines.extend(
        [
            "",
            "## V3 单次整页几何实验",
            "",
            "> 锚点召回表示模型红叉中心覆盖人工错题区域；事件召回使用题框与人工区域逐题匹配。仅红圈命中表示模型把真实错题区域错误列为circle-only。",
            "",
            "| 图片 | Prompt | 长边 | Schema | 红叉锚点 | 锚点真值命中 | 锚点真值召回 | 区域外锚点 | 错题事件 | 事件真值命中 | 事件真值召回 | 误报事件 | 仅红圈命中错题 | 重复事件 | 未分配锚点 | LLM请求 | 内容/OCR |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for summary in summaries:
        item = summary["checkpoints"]
        lines.append(
            "| {label} | {prompt} | {edge} | {schema} | {anchors} | {anchor_matched} | {anchor_recall} | {anchor_false} | {events} | {event_matched} | {event_recall} | {event_false} | {circle_false} | {duplicates} | {unassigned} | {requests} | {content} |".format(
                label=summary["label"],
                prompt=item["holistic_v3_prompt_version"],
                edge=item["holistic_v3_long_edge"],
                schema=item["holistic_v3_json_schema_status"],
                anchors=item["holistic_v3_cross_anchor_count"],
                anchor_matched=item[
                    "holistic_v3_cross_anchor_truth_matched_count"
                ],
                anchor_recall=item["holistic_v3_cross_anchor_truth_recall"],
                anchor_false=item["holistic_v3_cross_anchor_false_count"],
                events=item["holistic_v3_question_event_count"],
                event_matched=item[
                    "holistic_v3_question_event_truth_matched_count"
                ],
                event_recall=item["holistic_v3_question_event_truth_recall"],
                event_false=item["holistic_v3_question_event_false_count"],
                circle_false=item[
                    "holistic_v3_circle_only_false_positive_count"
                ],
                duplicates=item["holistic_v3_duplicate_event_count"],
                unassigned=item[
                    "holistic_v3_unassigned_cross_anchor_count"
                ],
                requests=item["holistic_v3_llm_request_count"],
                content=item["holistic_v3_content_ocr_status"],
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
        "| 图片 | 整页总耗时 | 红色证据CV | 红叉候选CV | 旧生产流程 | 旧stable实验 | 新方案总耗时 | LLM1输入 | LLM1核验次数 | 新方案LLM请求 | 定向复查触发 | 定向复查请求 | LLM1候选核验 | 独立漏检扫描 | fallback复核 | baseline LLM2 | 空间LLM2 | LLM2定位总计 | 定向复查耗时 | 后置审计 | Profile | 空间分组请求 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for summary in summaries:
        case_timings = summary.get("timings_ms") or {}
        checkpoints = summary["checkpoints"]
        stage_timings = checkpoints.get("cross_anchor_stage_timings_ms") or {}
        lines.append(
            "| {label} | {total} | {red_cv} | {cross_cv} | {production} | {stable} | {cross_anchor} | {input_mode} | {runs} | {requests} | {retry_triggers} | {retry_requests} | {verify} | {scan} | {fallback} | {baseline_llm2} | {spatial_llm2} | {llm2} | {llm2_second} | {audit} | {profile} | {spatial_requests} |".format(
                label=summary["label"],
                total=case_timings.get("total"),
                red_cv=case_timings.get("red_evidence_cv"),
                cross_cv=case_timings.get("cross_candidate_cv"),
                production=case_timings.get("production_pipeline"),
                stable=case_timings.get("stable_event_experiment"),
                cross_anchor=case_timings.get("cross_anchor_experiment"),
                input_mode=checkpoints.get("cross_anchor_llm1_input_mode"),
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
                baseline_llm2=stage_timings.get(
                    "baseline_llm2_localization"
                ),
                spatial_llm2=stage_timings.get("spatial_llm2_localization"),
                llm2=stage_timings.get("llm2_localization"),
                llm2_second=stage_timings.get(
                    "llm2_localization_run_002"
                ),
                audit=stage_timings.get("post_llm2_audit"),
                profile=checkpoints.get("cross_anchor_profile"),
                spatial_requests=checkpoints.get(
                    "cross_anchor_spatial_group_request_count"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## V3 单次整页几何耗时",
            "",
            "| 图片 | 输入长边 | 整页总耗时 | V3实验总耗时 | MiniMax请求耗时 | LLM请求 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        case_timings = summary.get("timings_ms") or {}
        checkpoints = summary["checkpoints"]
        holistic_timings = checkpoints.get("holistic_v3_stage_timings_ms") or {}
        lines.append(
            "| {label} | {edge} | {total} | {experiment} | {request} | {requests} |".format(
                label=summary["label"],
                edge=checkpoints.get("holistic_v3_long_edge"),
                total=case_timings.get("total"),
                experiment=case_timings.get("holistic_v3_experiment"),
                request=holistic_timings.get("model_request"),
                requests=checkpoints.get("holistic_v3_llm_request_count"),
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
        "--holistic-v3-only",
        action="store_true",
        help="run exactly one full-page geometry-only MiniMax request per image",
    )
    parser.add_argument(
        "--holistic-v3-long-edge",
        type=int,
        help="full-page input long edge for the isolated holistic V3 experiment",
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
        "--cross-anchor-confirmation-only",
        action="store_true",
        help=(
            "run CV candidate generation and LLM1 cross confirmation only; "
            "skip the production pipeline and LLM2 localization"
        ),
    )
    parser.add_argument(
        "--cross-anchor-profile",
        choices=(
            "baseline",
            "independent-rescue",
            "anchor-preserving",
            "spatial-grouped",
        ),
        default="baseline",
        help="select an isolated cross-anchor diagnostic experiment profile",
    )
    parser.add_argument(
        "--cross-anchor-replay-from",
        help=(
            "reuse LLM1 candidate verification, independent scan, and fallback "
            "verification from an earlier diagnostic output directory"
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

    if args.holistic_v3_only:
        if not args.cross_cv_config:
            parser.error("--holistic-v3-only requires --cross-cv-config")
        if not args.truth_regions:
            parser.error("--holistic-v3-only requires --truth-regions")
        if args.cv_only or args.cv_cross_only:
            parser.error("--holistic-v3-only cannot be combined with CV-only modes")
        if args.compare_stable_events or args.compare_cross_anchor:
            parser.error(
                "--holistic-v3-only cannot be combined with comparison modes"
            )
        if args.cross_anchor_replay_from:
            parser.error("--holistic-v3-only cannot be combined with replay")
    elif args.holistic_v3_long_edge is not None:
        parser.error("--holistic-v3-long-edge requires --holistic-v3-only")

    if args.cross_anchor_confirmation_only:
        if args.cv_only or args.cv_cross_only:
            parser.error(
                "--cross-anchor-confirmation-only cannot be combined with CV-only modes"
            )
        if args.holistic_v3_only or args.compare_stable_events:
            parser.error(
                "--cross-anchor-confirmation-only cannot be combined with other experiments"
            )
        if args.compare_cross_anchor:
            parser.error(
                "--cross-anchor-confirmation-only already selects the cross-anchor experiment"
            )

    if args.compare_cross_anchor or args.cross_anchor_confirmation_only:
        if not args.cross_cv_config:
            parser.error("cross-anchor experiment requires --cross-cv-config")
        if not args.truth_regions:
            parser.error("cross-anchor experiment requires --truth-regions")
    elif args.cross_anchor_replay_from:
        parser.error(
            "--cross-anchor-replay-from requires --compare-cross-anchor"
        )

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
    if (
        args.cv_cross_only
        or args.compare_cross_anchor
        or args.cross_anchor_confirmation_only
        or args.holistic_v3_only
    ):
        try:
            loader = (
                load_holistic_v3_inputs
                if args.holistic_v3_only
                else load_cross_cv_inputs
            )
            cross_cv_config, truth_by_label = loader(
                Path(args.cross_cv_config).expanduser().resolve(),
                Path(args.truth_regions).expanduser().resolve(),
                [label for label, _path in images],
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    holistic_v3_long_edge = None
    if args.holistic_v3_only:
        holistic_v3_long_edge = (
            args.holistic_v3_long_edge
            if args.holistic_v3_long_edge is not None
            else int(cross_cv_config["holistic_v3_default_long_edge"])
        )
        supported_long_edges = cross_cv_config["holistic_v3_supported_long_edges"]
        if holistic_v3_long_edge not in supported_long_edges:
            parser.error(
                "--holistic-v3-long-edge must be one of "
                + ", ".join(str(value) for value in supported_long_edges)
            )
    output_dir = Path(args.output).expanduser().resolve()
    cross_anchor_replay_from = (
        Path(args.cross_anchor_replay_from).expanduser().resolve()
        if args.cross_anchor_replay_from
        else None
    )
    if cross_anchor_replay_from is not None and not cross_anchor_replay_from.is_dir():
        parser.error("--cross-anchor-replay-from must be an existing directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        output_dir / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "warning": "Contains worksheet images, prompts, and model responses; delete after analysis.",
            "cv_only": args.cv_only or args.cv_cross_only,
            "cv_cross_only": args.cv_cross_only,
            "holistic_v3_only": args.holistic_v3_only,
            "holistic_v3_long_edge": holistic_v3_long_edge,
            "compare_stable_events": args.compare_stable_events,
            "compare_cross_anchor": args.compare_cross_anchor,
            "cross_anchor_confirmation_only": args.cross_anchor_confirmation_only,
            "cross_anchor_profile": args.cross_anchor_profile,
            "cross_anchor_replay_from": (
                str(cross_anchor_replay_from)
                if cross_anchor_replay_from is not None
                else None
            ),
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
            compare_cross_anchor=(
                args.compare_cross_anchor or args.cross_anchor_confirmation_only
            ),
            cross_anchor_confirmation_only=args.cross_anchor_confirmation_only,
            holistic_v3_only=args.holistic_v3_only,
            holistic_v3_long_edge=holistic_v3_long_edge,
            cross_anchor_profile=args.cross_anchor_profile,
            cross_anchor_replay_from=cross_anchor_replay_from,
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
            or summary.get("holistic_v3_experiment_error")
            for summary in summaries
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
