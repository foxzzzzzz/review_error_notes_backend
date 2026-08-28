"""MiniMax Token Plan image understanding and structured recognition."""

from __future__ import annotations

import base64
import io
import inspect
import json
import logging
import math
import re
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Literal, Optional

import httpx
from PIL import Image, ImageOps
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.config import settings
from app.services.error_mark_validation import (
    ErrorMarkImageInvalid,
    RedMarkScanResult,
    filter_valid_error_marks,
    normalize_error_mark_groups,
    validate_localization_red_evidence,
)
from app.services.question_collection import collection_reason_for, collection_status_for
from app.services.recognition_policy import decide_candidate
from app.services.tag_normalization import normalize_tags


VISION_PATH = "/v1/coding_plan/vlm"
OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
FORMAT_RETRYABLE_ERROR_CODES = {
    "vision_response_empty",
    "vision_response_json_invalid",
    "vision_response_schema_invalid",
}
FORMAT_RETRY_INSTRUCTION = (
    "\n\n格式纠偏：上次返回无法通过结构化解析。只返回严格 JSON，"
    "不要解释，不要 Markdown，不要在 JSON 前后添加任何文字。"
)
logger = logging.getLogger(__name__)


def recognition_correction_instruction(correction: Optional[str]) -> str:
    instructions = {
        "missed_errors": "本图上次可能漏识别错题：请重点复查红色错误标记及其对应的完整作答单元。",
        "false_positives": "本图上次可能误识别正确题：只保留具有可靠错误证据的题目，不要把无错误标记的正确题列为错题。",
        "both": "本图上次可能同时漏识别错题并误识别正确题：重点复查红色错误标记及其对应作答单元，并只保留具有可靠错误证据的题目。",
    }
    return instructions.get(correction, "")

RECOGNITION_PROMPT = """你是小学错题内容与红色批改标记识别器。请观察整张图片，一次性输出题目内容和独立的红色错误标记。

要求：
1. 忽略 Date、日期栏、页码、装订线、空白横线和与错题无关的印刷页眉。
2. 识别老师用于标识错误的红圈、红叉、红色删除线、红色波浪线、红色下划线或红色纠错批注；不要把印刷装饰色或单独的红色对勾误判为错误标记。
3. 一个 error_mark 表示老师的一次判错事件；同一作答单元上的红圈、红叉和纠正笔迹视为同一标记组。同一作答单元上的邻近红×与红圈必须合并为一个 cross_circle 标记，不得分别输出；红×写入 cross_bbox，红圈写入 circle_bbox，bbox 覆盖二者联合范围。
4. 只有红×、只有红圈或其他批改形式时不得伪造不存在的组成部分。每个判错事件只输出一个 error_mark，mark_id 从 0 开始连续且唯一。
5. error_mark 的 bbox、cross_bbox 和 circle_bbox 使用归一化角点格式 [left, top, right, bottom]，满足 0 <= left < right <= 1 和 0 <= top < bottom <= 1。
6. 题目 item 与 error_marks 必须分别识别。题目 item 不得输出 bbox，也不得预先绑定 mark_id；后续步骤会根据整图内容独立完成匹配和定位。
7. item 的粒度必须是最小可独立作答单元，不能把整道编号大题合并成一个 item。拼音格、完整词语格组、单个填空、单个选择项或一道计算题分别视为独立作答单元。
8. 若存在明确的红色错误标记，每个被标记的独立作答单元输出一个 item；同一道编号大题中有多个被标记的小题时，必须分别输出多个 item。同一行存在多个兄弟小题时，不得加入未标记的兄弟小题。
9. 对于看词语写拼音、看拼音写词语等词语练习，最小可独立作答单元是完整词语格组。即使标记只覆盖单字、单音节或部分笔画，也要识别其所属完整词语；完整词语优先于红色标记的像素覆盖范围。
10. 完整词语格组的各字段必须保持同一范围：raw_text 抄录学生对整个词语的实际作答，prompt_text 填写整个印刷提示，answer 填写整个正确答案，question_type 根据完整提示与完整作答判断。
11. 所有字段必须从当前图片可见内容提取，不得复用提示中的示例或臆造图片中不存在的词语。
12. 本次是有红标作业识别：只输出与可靠红色错误标记关联的最小作答单元；禁止在没有可靠红标关联时回退输出整页题目。
13. 红色批改符号本身不要写入 raw_text；老师写出的纠正内容可作为 answer 的参考。
14. raw_text 必须忠实抄录学生实际书写，包括错字、漏字、错误拼音和错误答案；禁止自动改正后覆盖原文。题目未作答时必须输出 "raw_text": ""，不得遗漏该字段或编造内容。
15. instruction 必须填写图片中可见的原始练习要求；prompt_text 必须填写重新出卷时展示的干净提示材料。二者不得包含学生作答、正确答案或老师批改笔迹。
16. question_type 只能是 write_pinyin、write_word、fill_blank、calculation、other 之一。无法确认的内容保留原样并写入 uncertain_segments。
17. confidence 范围为 0 到 1。difficulty 必须是 1 到 5 的整数，1 表示很简单，5 表示很难。
18. tags 只能使用中文标签，例如“拼音”“词语”“错别字”“老师批改”，不得返回 pinyin、word、teacher-marked 等英文编码。
19. 只输出一个 JSON 对象，不要解释，不要 Markdown。

JSON 格式：
{{
  "items": [{{
    "raw_text": "学生实际书写",
    "instruction": "原练习要求",
    "prompt_text": "不含学生作答的干净提示材料",
    "normalized_text": "规范内容或 null",
    "answer": "正确答案或 null",
    "subject": "math|chinese|english",
    "question_type": "write_pinyin|write_word|fill_blank|calculation|other",
    "tags": ["中文标签"],
    "difficulty": 1,
    "confidence": 0.95,
    "uncertain_segments": []
  }}],
  "error_marks": [{{
    "mark_id": 0,
    "mark_type": "cross_circle|circle|cross|deletion|underline|annotation|mixed",
    "bbox": [0.0, 0.0, 1.0, 1.0],
    "cross_bbox": [0.0, 0.0, 1.0, 1.0],
    "circle_bbox": [0.0, 0.0, 1.0, 1.0],
    "confidence": 0.95
  }}],
  "ignored_text": ["被忽略的页眉"]
}}

cross_bbox 或 circle_bbox 不存在时，对应字段必须返回 null。

科目提示：{subject_hint}
"""


def recognition_prompt_for(
    mode: Literal["marked", "unmarked"],
    subject_hint: Optional[str],
    local_red_regions: List[List[float]],
    correction: Optional[str] = None,
) -> str:
    """Build a mode-specific prompt from bounded local evidence."""
    if mode == "marked":
        mode_instruction = (
            "本地红标扫描发现以下归一化候选区域："
            + json.dumps(local_red_regions[:50], ensure_ascii=False, separators=(",", ":"))
            + "。这些区域只是定位提示，仍须按图片核验；先识别老师的一次判错事件。"
            + "邻近红×与红圈属于同一次批改时必须合并为一个 mark，不得分别输出；"
            + "分别填写 cross_bbox 和 circle_bbox。只输出与可靠红色错误标记关联的作答单元，禁止输出整页未标记题目。"
        )
    else:
        mode_instruction = (
            "本地未检测到可靠红色批改标记。本次按无红标作业分析：输出图片中的所有最小可独立作答单元，"
            "分别忠实提取学生答案与参考答案供后端比较；不得把所有输出单元直接称为错题。"
        )
    correction_instruction = recognition_correction_instruction(correction)
    base_prompt = RECOGNITION_PROMPT
    if mode == "unmarked":
        base_prompt = base_prompt.replace(
            "本次是有红标作业识别：只输出与可靠红色错误标记关联的最小作答单元；禁止在没有可靠红标关联时回退输出整页题目。",
            "本次是无红标作业分析：输出图片中的所有最小可独立作答单元，但不得把所有输出单元直接判为错题。",
        )
    return (
        base_prompt.format(subject_hint=subject_hint or "自动判断")
        + "\n\n识别模式要求："
        + mode_instruction
        + ("\n\n纠偏要求：" + correction_instruction if correction_instruction else "")
    )

LOCALIZATION_PROMPT = """你是小学错题区域定位器。请根据整张图片、已识别题目内容和独立红色错误标记，重新完成题目与标记匹配及作答单元定位。

要求：
1. 必须在整张图片中独立定位，不存在任何第一次题目区域可供参考。只能依据当前图片可见的印刷提示、学生答案和红色错误标记作出判断。
2. 对每个题目 index，先核对其 prompt_text、raw_text、instruction，再选择属于该题的 error_marks，并把对应 mark_id 写入 mark_ids。
3. matched=true 时，bbox 必须覆盖该题完整的印刷提示、学生答案和相关红色批改标记，是可独立理解和再次作答的最小完整单元。
4. bbox 不得包含未标记的相邻兄弟小题，不得把一个题目的区域或标记配给另一个题目，同一 mark_id 不得分配给多个题目。
5. 在 observed_prompt_text 和 observed_raw_text 中忠实抄录 bbox 内实际看到的提示与学生答案，用于后端核对；不得照抄输入值来代替观察。
6. 找不到对应可见内容、标记无法归属、内容与输入题目矛盾或无法确认时，必须返回 matched=false、mark_ids=[]、bbox=null、observed_prompt_text=null、observed_raw_text=null。
7. bbox 使用归一化角点格式 [left, top, right, bottom]。
8. 对下面每个 index 恰好返回一次，不得缺失、重复或新增 index。
9. confidence 表示内容匹配、标记归属和完整区域定位的综合置信度，范围 0 到 1。
10. 只输出 JSON，不要解释，不要 Markdown。

输入数据：
__INPUT__

返回格式：
{"items":[{"index":0,"matched":true,"mark_ids":[0],"bbox":[0.0,0.0,1.0,1.0],"observed_prompt_text":"图片内实际提示","observed_raw_text":"图片内实际作答","confidence":0.95}]}
"""


class VisionRecognitionError(RuntimeError):
    """Safe recognition error that never contains credentials or image bytes."""

    def __init__(
        self,
        code: str,
        user_message: str | None = None,
        diagnostic: dict | None = None,
    ):
        if user_message is None:
            code = "vision_response_invalid"
            user_message = "识别结果异常，请稍后重试"
        self.code = code
        self.user_message = user_message
        self.diagnostic = diagnostic or {}
        super().__init__(user_message)


class ImageReviewRequired(VisionRecognitionError):
    """A safe, actionable image-level issue rather than a processing failure."""


SAFE_RECOGNITION_DIAGNOSTIC_KEYS = {
    "operation",
    "status_code",
    "provider_status_code",
    "validation_fields",
    "reason",
    "source_width",
    "source_height",
    "prepared_width",
    "prepared_height",
    "candidate_count",
    "mark_count",
    "localization_status",
    "localization_error_code",
    "localization_error_reason",
    "localization_error_diagnostic",
    "localization_returned_count",
    "localization_validated_count",
    "localization_verified_count",
    "localization_reliable_mark_count",
    "localization_rejection_counts",
    "localization_geometry_failure_counts",
    "localization_geometry_diagnostics",
    "localization_assigned_mark_ids",
    "localization_unassigned_mark_ids",
    "localization_unassigned_mark_count",
    "localization_missing_mark_diagnostics",
    "response_content_length",
    "json_error_position",
    "json_error_line",
    "json_error_column",
    "has_markdown_fence",
    "has_output_wrapper",
    "first_non_whitespace_char_type",
    "last_non_whitespace_char_type",
    "likely_truncated",
    "parsed_json_type",
    "response_attempt",
    "response_max_attempts",
}


def safe_recognition_diagnostic(error: Exception) -> dict:
    if not isinstance(error, VisionRecognitionError):
        return {}
    return {
        key: value
        for key, value in error.diagnostic.items()
        if key in SAFE_RECOGNITION_DIAGNOSTIC_KEYS
    }


def validate_normalized_bbox(value: List[float]) -> List[float]:
    if len(value) != 4 or any(coordinate < 0 or coordinate > 1 for coordinate in value):
        raise ValueError("bbox must contain four normalized coordinates")
    left, top, right, bottom = value
    if left >= right or top >= bottom:
        raise ValueError("bbox must contain ordered left, top, right, bottom coordinates")
    return value


class VisionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    raw_text: str
    instruction: str
    prompt_text: str
    normalized_text: Optional[str] = None
    answer: Optional[str] = None
    subject: Literal["math", "chinese", "english"]
    question_type: Literal["write_pinyin", "write_word", "fill_blank", "calculation", "other"]
    tags: List[str] = Field(default_factory=list)
    difficulty: int = Field(default=3, ge=1, le=5)
    confidence: float = Field(ge=0, le=1)
    uncertain_segments: List[str] = Field(default_factory=list)

    @field_validator("difficulty", mode="before")
    @classmethod
    def clamp_integer_difficulty_to_supported_range(cls, value):
        if isinstance(value, int) and not isinstance(value, bool):
            return max(1, min(5, value))
        return value

    @field_validator("instruction", "prompt_text")
    @classmethod
    def text_fields_must_not_be_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("text field must not be blank")
        return value

class ErrorMark(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mark_id: int = Field(ge=0)
    mark_type: Literal[
        "circle",
        "cross",
        "cross_circle",
        "deletion",
        "underline",
        "annotation",
        "mixed",
    ]
    bbox: List[float]
    cross_bbox: Optional[List[float]] = None
    circle_bbox: Optional[List[float]] = None
    confidence: float = Field(ge=0, le=1)

    @field_validator("bbox", "cross_bbox", "circle_bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        return validate_normalized_bbox(value) if value is not None else None


class VisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: List[VisionItem] = Field(min_length=1)
    error_marks: List[ErrorMark] = Field(default_factory=list)
    ignored_text: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def mark_ids_must_be_unique_and_sequential(self):
        mark_ids = [mark.mark_id for mark in self.error_marks]
        if mark_ids != list(range(len(mark_ids))):
            raise ValueError("error mark ids must be unique and sequential")
        return self


class LocalizationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int = Field(ge=0)
    matched: bool
    mark_ids: List[int] = Field(default_factory=list)
    bbox: Optional[List[float]] = None
    observed_prompt_text: Optional[str] = None
    observed_raw_text: Optional[str] = None
    confidence: float = Field(ge=0, le=1)

    @field_validator("bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        if value is None:
            return value
        return validate_normalized_bbox(value)

    @model_validator(mode="after")
    def matched_result_must_have_bbox(self):
        if self.matched != (self.bbox is not None):
            raise ValueError("matched localization must have bbox and unmatched must not")
        if not self.matched and self.mark_ids:
            raise ValueError("unmatched localization cannot assign marks")
        if not self.matched and (
            self.observed_prompt_text is not None or self.observed_raw_text is not None
        ):
            raise ValueError("unmatched localization cannot contain observed evidence")
        return self


class LocalizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: List[LocalizationItem] = Field(min_length=1)


def validated_localizations(
    result: LocalizationResult,
    item_count: int,
    marks: dict[int, ErrorMark],
) -> dict[int, LocalizationItem]:
    def localization_error(reason: str) -> VisionRecognitionError:
        return VisionRecognitionError(
            "vision_localization_invalid",
            "题目定位结果不完整，请稍后重试",
            diagnostic={
                "operation": "localization",
                "reason": reason,
                "item_count": item_count,
                "mark_count": len(marks),
            },
        )

    indexes = [item.index for item in result.items]
    if len(indexes) != item_count or len(set(indexes)) != len(indexes):
        raise localization_error("index_count_mismatch")
    if set(indexes) != set(range(item_count)):
        raise localization_error("index_set_mismatch")
    assigned_mark_ids = [
        mark_id
        for item in result.items
        for mark_id in item.mark_ids
    ]
    if len(assigned_mark_ids) != len(set(assigned_mark_ids)):
        raise localization_error("duplicate_mark_assignment")
    if set(assigned_mark_ids) - set(marks):
        raise localization_error("unknown_mark_assignment")
    return {item.index: item for item in result.items}


def bbox_area(bbox: List[float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_contains_center(container: List[float], candidate: List[float]) -> bool:
    center_x = (candidate[0] + candidate[2]) / 2
    center_y = (candidate[1] + candidate[3]) / 2
    return (
        container[0] <= center_x <= container[2]
        and container[1] <= center_y <= container[3]
    )


def bboxes_intersect(first: List[float], second: List[float]) -> bool:
    return not (
        first[2] < second[0]
        or second[2] < first[0]
        or first[3] < second[1]
        or second[3] < first[1]
    )


def mark_distance_diagnostic(
    localization_bbox: List[float],
    mark_id: int,
    mark_bbox: List[float],
) -> dict:
    center_x = (mark_bbox[0] + mark_bbox[2]) / 2
    center_y = (mark_bbox[1] + mark_bbox[3]) / 2
    horizontal_gap = max(
        localization_bbox[0] - center_x,
        0.0,
        center_x - localization_bbox[2],
    )
    vertical_gap = max(
        localization_bbox[1] - center_y,
        0.0,
        center_y - localization_bbox[3],
    )
    return {
        "mark_id": mark_id,
        "horizontal_gap_ratio": round(horizontal_gap, 6),
        "vertical_gap_ratio": round(vertical_gap, 6),
        "nearest_distance_ratio": round(
            math.hypot(horizontal_gap, vertical_gap),
            6,
        ),
        "mark_bbox_intersects_question_bbox": bboxes_intersect(
            localization_bbox,
            mark_bbox,
        ),
    }


def mark_anchor_bbox(mark: ErrorMark) -> tuple[List[float], str]:
    """Return the correction component that should anchor a question crop."""
    if mark.mark_type == "cross_circle" and mark.circle_bbox is not None:
        return mark.circle_bbox, "circle"
    if mark.mark_type == "cross" and mark.cross_bbox is not None:
        return mark.cross_bbox, "cross"
    return mark.bbox, mark.mark_type


def mark_anchor_diagnostic(
    localization_bbox: List[float],
    mark_id: int,
    mark: ErrorMark,
    max_gap_ratio: float,
) -> dict:
    anchor_bbox, anchor_type = mark_anchor_bbox(mark)
    horizontal_gap = max(
        localization_bbox[0] - anchor_bbox[2],
        0.0,
        anchor_bbox[0] - localization_bbox[2],
    )
    vertical_gap = max(
        localization_bbox[1] - anchor_bbox[3],
        0.0,
        anchor_bbox[1] - localization_bbox[3],
    )
    distance = math.hypot(horizontal_gap, vertical_gap)
    intersects = bboxes_intersect(localization_bbox, anchor_bbox)
    return {
        "mark_id": mark_id,
        "mark_type": mark.mark_type,
        "anchor_type": anchor_type,
        "anchor_bbox": list(anchor_bbox),
        "horizontal_gap_ratio": round(horizontal_gap, 6),
        "vertical_gap_ratio": round(vertical_gap, 6),
        "nearest_distance_ratio": round(distance, 6),
        "max_gap_ratio": max_gap_ratio,
        "intersects_question_bbox": intersects,
        "accepted": intersects or distance <= max_gap_ratio,
    }


def nearest_red_region_diagnostic(
    localization_bbox: List[float],
    red_regions,
) -> dict | None:
    candidates = []
    for region_index, region in enumerate(red_regions):
        region_bbox = region.bbox
        horizontal_gap = max(
            localization_bbox[0] - region_bbox[2],
            0.0,
            region_bbox[0] - localization_bbox[2],
        )
        vertical_gap = max(
            localization_bbox[1] - region_bbox[3],
            0.0,
            region_bbox[1] - localization_bbox[3],
        )
        distance = math.hypot(horizontal_gap, vertical_gap)
        candidates.append(
            (
                distance,
                region_index,
                {
                    "region_index": region_index,
                    "horizontal_gap_ratio": round(horizontal_gap, 6),
                    "vertical_gap_ratio": round(vertical_gap, 6),
                    "nearest_distance_ratio": round(distance, 6),
                    "intersects_question_bbox": bboxes_intersect(
                        localization_bbox,
                        region_bbox,
                    ),
                    "region_area_ratio": region.area_ratio,
                    "region_pixel_count": region.pixel_count,
                },
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]


def safe_local_red_validation(diagnostic: dict | None) -> dict | None:
    if diagnostic is None:
        return None
    safe_keys = (
        "accepted",
        "reason",
        "red_pixel_count",
        "pixel_count",
        "red_pixel_ratio",
        "red_pixel_min_ratio",
        "expansion_ratio",
    )
    return {key: diagnostic.get(key) for key in safe_keys if key in diagnostic}


def marker_focused_display_bbox(
    localization_bbox: List[float],
    mark_ids: List[int],
    marks: dict[int, ErrorMark],
    padding_ratio: float,
) -> List[float]:
    if padding_ratio == 0:
        return list(localization_bbox)

    left, top, right, bottom = localization_bbox
    content_width = right - left
    content_height = bottom - top
    display_width = min(1.0, content_width * (1 + 2 * padding_ratio))
    display_height = min(1.0, content_height * (1 + 2 * padding_ratio))

    assigned_marks = [marks[mark_id] for mark_id in mark_ids if mark_id in marks]
    if assigned_marks:
        focus_x = sum(
            (mark.bbox[0] + mark.bbox[2]) / 2 for mark in assigned_marks
        ) / len(assigned_marks)
        focus_y = sum(
            (mark.bbox[1] + mark.bbox[3]) / 2 for mark in assigned_marks
        ) / len(assigned_marks)
    else:
        focus_x = (left + right) / 2
        focus_y = (top + bottom) / 2

    min_center_x = max(display_width / 2, right - display_width / 2)
    max_center_x = min(1 - display_width / 2, left + display_width / 2)
    min_center_y = max(display_height / 2, bottom - display_height / 2)
    max_center_y = min(1 - display_height / 2, top + display_height / 2)
    center_x = min(max(focus_x, min_center_x), max_center_x)
    center_y = min(max(focus_y, min_center_y), max_center_y)

    return [
        max(0.0, center_x - display_width / 2),
        max(0.0, center_y - display_height / 2),
        min(1.0, center_x + display_width / 2),
        min(1.0, center_y + display_height / 2),
    ]


def localization_passes_geometry(
    localization: LocalizationItem,
    marks: dict[int, ErrorMark],
    max_area_ratio: float,
    allow_unassigned_marks: bool = False,
    anchor_max_gap_ratio: float = 0.0,
    cross_only_max_gap_ratio: float = 0.0,
) -> bool:
    return localization_geometry_diagnostic(
        localization,
        marks=marks,
        max_area_ratio=max_area_ratio,
        allow_unassigned_marks=allow_unassigned_marks,
        anchor_max_gap_ratio=anchor_max_gap_ratio,
        cross_only_max_gap_ratio=cross_only_max_gap_ratio,
    )["passed"]


def localization_geometry_diagnostic(
    localization: LocalizationItem,
    marks: dict[int, ErrorMark],
    max_area_ratio: float,
    allow_unassigned_marks: bool = False,
    anchor_max_gap_ratio: float = 0.0,
    cross_only_max_gap_ratio: float = 0.0,
) -> dict:
    """Explain geometry rejection without exposing recognized worksheet text."""
    failure_reasons = []
    area_ratio = (
        bbox_area(localization.bbox) if localization.bbox is not None else None
    )
    if not localization.matched or localization.bbox is None:
        failure_reasons.append("not_matched_or_missing_bbox")
    if area_ratio is not None and area_ratio > max_area_ratio:
        failure_reasons.append("bbox_area_exceeded")
    if marks and not localization.mark_ids and not allow_unassigned_marks:
        failure_reasons.append("missing_mark_ids")

    missing_mark_ids = [
        mark_id for mark_id in localization.mark_ids if mark_id not in marks
    ]
    if missing_mark_ids:
        failure_reasons.append("unknown_mark_ids")
    use_tiered_anchors = anchor_max_gap_ratio > 0 or cross_only_max_gap_ratio > 0
    anchor_diagnostics = []
    if localization.bbox is not None and use_tiered_anchors:
        for mark_id in localization.mark_ids:
            if mark_id not in marks:
                continue
            mark = marks[mark_id]
            max_gap_ratio = (
                cross_only_max_gap_ratio
                if mark.mark_type == "cross"
                else anchor_max_gap_ratio
            )
            anchor_diagnostics.append(
                mark_anchor_diagnostic(
                    localization.bbox,
                    mark_id,
                    mark,
                    max_gap_ratio,
                )
            )
        outside_mark_ids = [
            diagnostic["mark_id"]
            for diagnostic in anchor_diagnostics
            if not diagnostic["accepted"]
        ]
    else:
        outside_mark_ids = [
            mark_id
            for mark_id in localization.mark_ids
            if localization.bbox is not None
            and mark_id in marks
            and not bbox_contains_center(localization.bbox, marks[mark_id].bbox)
        ]
    if outside_mark_ids:
        failure_reasons.append(
            "mark_anchor_too_far"
            if use_tiered_anchors
            else "mark_center_outside_bbox"
        )
    outside_mark_diagnostics = (
        [
            diagnostic
            for diagnostic in anchor_diagnostics
            if not diagnostic["accepted"]
        ]
        if use_tiered_anchors
        else [
            mark_distance_diagnostic(
                localization.bbox,
                mark_id,
                marks[mark_id].bbox,
            )
            for mark_id in outside_mark_ids
            if localization.bbox is not None
        ]
    )

    diagnostic = {
        "passed": not failure_reasons,
        "bbox_area_ratio": round(area_ratio, 6) if area_ratio is not None else None,
        "max_area_ratio": max_area_ratio,
        "mark_ids": list(localization.mark_ids),
        "missing_mark_ids": missing_mark_ids,
        "outside_mark_ids": outside_mark_ids,
        "outside_mark_diagnostics": outside_mark_diagnostics,
        "failure_reasons": failure_reasons,
    }
    if use_tiered_anchors:
        diagnostic["anchor_diagnostics"] = anchor_diagnostics
    return diagnostic


def _normalized_evidence_text(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).lower()
    return "".join(character for character in normalized if character.isalnum())


def localization_matches_evidence(
    localization: LocalizationItem,
    item: VisionItem,
) -> bool:
    expected_observed_pairs = (
        (item.prompt_text, localization.observed_prompt_text),
        (item.raw_text, localization.observed_raw_text),
    )
    return any(
        _normalized_evidence_text(expected)
        and _normalized_evidence_text(expected) == _normalized_evidence_text(observed)
        for expected, observed in expected_observed_pairs
    )


def repair_unique_mark_assignments(
    localizations: dict[int, LocalizationItem],
    marks: dict[int, ErrorMark],
    items: List[VisionItem],
    *,
    localization_threshold: float,
    max_area_ratio: float,
    anchor_max_gap_ratio: float,
    cross_only_max_gap_ratio: float,
) -> tuple[dict[int, LocalizationItem], List[dict]]:
    """Assign an unused correction event only when both sides have one best match."""
    repaired = dict(localizations)
    assigned_mark_ids = {
        mark_id
        for localization in localizations.values()
        for mark_id in localization.mark_ids
    }
    eligible_localizations = {
        index: localization
        for index, localization in localizations.items()
        if 0 <= index < len(items)
        and localization.matched
        and localization.bbox is not None
        and not localization.mark_ids
        and localization.confidence >= localization_threshold
        and bbox_area(localization.bbox) <= max_area_ratio
        and localization_matches_evidence(localization, items[index])
    }
    available_marks = {
        mark_id: mark
        for mark_id, mark in marks.items()
        if mark_id not in assigned_mark_ids
    }

    edges = []
    for index, localization in eligible_localizations.items():
        for mark_id, mark in available_marks.items():
            max_gap_ratio = (
                cross_only_max_gap_ratio
                if mark.mark_type == "cross"
                else anchor_max_gap_ratio
            )
            diagnostic = mark_anchor_diagnostic(
                localization.bbox,
                mark_id,
                mark,
                max_gap_ratio,
            )
            if diagnostic["accepted"]:
                edges.append(
                    (
                        diagnostic["nearest_distance_ratio"],
                        index,
                        mark_id,
                        diagnostic,
                    )
                )

    def unique_best(candidates, identity_position):
        if not candidates:
            return None
        ordered = sorted(candidates, key=lambda candidate: (candidate[0], candidate[identity_position]))
        if len(ordered) > 1 and math.isclose(
            ordered[0][0],
            ordered[1][0],
            abs_tol=1e-9,
        ):
            return None
        return ordered[0]

    best_by_index = {
        index: unique_best(
            [edge for edge in edges if edge[1] == index],
            2,
        )
        for index in eligible_localizations
    }
    best_by_mark = {
        mark_id: unique_best(
            [edge for edge in edges if edge[2] == mark_id],
            1,
        )
        for mark_id in available_marks
    }

    repair_diagnostics = []
    for index in sorted(best_by_index):
        edge = best_by_index[index]
        if edge is None:
            continue
        _, _, mark_id, anchor_diagnostic = edge
        reverse_edge = best_by_mark.get(mark_id)
        if reverse_edge is None or reverse_edge[1] != index:
            continue
        repaired[index] = repaired[index].model_copy(update={"mark_ids": [mark_id]})
        repair_diagnostics.append(
            {
                "index": index,
                "mark_id": mark_id,
                "assignment_source": "deterministic",
                "anchor_type": anchor_diagnostic["anchor_type"],
                "nearest_distance_ratio": anchor_diagnostic[
                    "nearest_distance_ratio"
                ],
            }
        )
    return repaired, repair_diagnostics


def build_question_values(
    item: VisionItem,
    index: int,
    confidence_threshold: float,
    localization: Optional[LocalizationItem],
    localization_threshold: float,
    localization_max_area_ratio: float,
    marks: dict[int, ErrorMark],
    normalized_tags: List[str],
    crop_context_padding_ratio: float = 0.0,
    localization_red_verified: bool = False,
    anchor_max_gap_ratio: float = 0.0,
    cross_only_max_gap_ratio: float = 0.0,
    circle_only_requires_review: bool = False,
) -> dict:
    """Map a validated vision item to the existing question persistence contract."""
    localization_present = localization is not None
    localization_matched = bool(localization and localization.matched)
    localization_has_bbox = bool(localization and localization.bbox is not None)
    localization_confidence_passed = bool(
        localization and localization.confidence >= localization_threshold
    )
    geometry_diagnostic = (
        localization_geometry_diagnostic(
            localization,
            marks=marks,
            max_area_ratio=localization_max_area_ratio,
            allow_unassigned_marks=localization_red_verified,
            anchor_max_gap_ratio=anchor_max_gap_ratio,
            cross_only_max_gap_ratio=cross_only_max_gap_ratio,
        )
        if localization is not None
        else None
    )
    localization_geometry_passed = bool(
        geometry_diagnostic and geometry_diagnostic["passed"]
    )
    localization_text_evidence_passed = bool(
        localization and localization_matches_evidence(localization, item)
    )
    assigned_marks = [
        marks[mark_id]
        for mark_id in (localization.mark_ids if localization else [])
        if mark_id in marks
    ]
    circle_only_evidence = circle_only_requires_review and bool(assigned_marks) and all(
        mark.mark_type == "circle" for mark in assigned_marks
    )
    localization_verified = (
        localization_present
        and localization_matched
        and localization_has_bbox
        and localization_confidence_passed
        and localization_geometry_passed
        and localization_text_evidence_passed
    )
    needs_review = (
        item.confidence < confidence_threshold
        or bool(item.uncertain_segments)
        or not localization_verified
        or circle_only_evidence
    )
    crop_region = {
        "bbox_source": "unverified",
        "localization_status": "needs_review",
        "index": index,
    }
    if localization_verified:
        bbox_source = (
            "local_red_verified"
            if localization_red_verified and not localization.mark_ids
            else "minimax_marker_anchored"
        )
        display_bbox = marker_focused_display_bbox(
            localization_bbox=localization.bbox,
            mark_ids=localization.mark_ids,
            marks=marks,
            padding_ratio=crop_context_padding_ratio,
        )
        crop_region = {
            "bbox": display_bbox,
            "bbox_format": "normalized_ltrb",
            "bbox_source": bbox_source,
            "bbox_confidence": localization.confidence,
            "localization_status": "verified",
            "mark_ids": localization.mark_ids,
            "index": index,
        }
        if crop_context_padding_ratio > 0:
            crop_region.update(
                {
                    "localization_bbox": localization.bbox,
                    "bbox_source": (
                        "local_red_verified_context"
                        if localization_red_verified
                        and not localization.mark_ids
                        else "marker_focused_context"
                    ),
                    "display_context_padding_ratio": crop_context_padding_ratio,
                }
            )
    reliable_error_mark = bool(
        localization_verified
        and (localization.mark_ids or localization_red_verified)
    )
    values = {
        "crop_region": crop_region,
        "subject": item.subject,
        "ocr_text": item.raw_text,
        "ocr_answer": item.answer,
        "ocr_raw_json": {"provider": "minimax", **item.model_dump(mode="json")},
        "question_type": item.question_type,
        "tags": normalized_tags,
        "difficulty": item.difficulty,
        "review_status": "needs_review" if needs_review else "confirmed",
    }
    values["collection_status"] = collection_status_for(
        SimpleNamespace(**values, reliable_error_mark=reliable_error_mark)
    )
    values["ocr_raw_json"]["collection_reason"] = collection_reason_for(
        SimpleNamespace(**values, reliable_error_mark=reliable_error_mark)
    )
    values["ocr_raw_json"]["reliable_error_mark"] = reliable_error_mark
    values["ocr_raw_json"]["localization_validation"] = {
        "present": localization_present,
        "matched": localization_matched,
        "has_bbox": localization_has_bbox,
        "confidence_passed": localization_confidence_passed,
        "geometry_passed": localization_geometry_passed,
        "text_evidence_passed": localization_text_evidence_passed,
        "local_red_verified": localization_red_verified,
        "circle_only_evidence": circle_only_evidence,
        "verified": localization_verified,
        "geometry": geometry_diagnostic,
    }
    return values


def _recognize_with_mode(
    client,
    image_path: str,
    subject_hint: Optional[str],
    recognition_correction: Optional[str],
    mode: Literal["marked", "unmarked"],
    local_red_regions: List[List[float]],
) -> VisionResult:
    candidate_kwargs = {
        "subject_hint": subject_hint,
        "recognition_correction": recognition_correction,
        "recognition_mode": mode,
        "local_red_regions": local_red_regions,
    }
    supported = inspect.signature(client.recognize).parameters
    kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in supported and value is not None
    }
    return client.recognize(image_path, **kwargs)


def _localize_with_correction(
    client,
    image_path: str,
    items: List[VisionItem],
    marks: List[ErrorMark],
    correction: Optional[dict] = None,
) -> LocalizationResult:
    supported = inspect.signature(client.localize).parameters
    kwargs = {"correction": correction} if "correction" in supported else {}
    return client.localize(image_path, items, marks, **kwargs)


def localization_semantic_diagnostic(
    localizations: dict[int, LocalizationItem],
    marks: dict[int, ErrorMark],
    *,
    max_area_ratio: float,
    anchor_max_gap_ratio: float,
    cross_only_max_gap_ratio: float,
) -> dict:
    reason_counts: dict[str, int] = {}
    failed_items = []
    passed_count = 0
    for index, localization in sorted(localizations.items()):
        diagnostic = localization_geometry_diagnostic(
            localization,
            marks,
            max_area_ratio,
            anchor_max_gap_ratio=anchor_max_gap_ratio,
            cross_only_max_gap_ratio=cross_only_max_gap_ratio,
        )
        if diagnostic["passed"]:
            passed_count += 1
            continue
        for reason in diagnostic["failure_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        failed_items.append(
            {
                "index": index,
                "mark_ids": list(localization.mark_ids),
                "failure_reasons": list(diagnostic["failure_reasons"]),
            }
        )
    assigned_mark_ids = sorted(
        {
            mark_id
            for localization in localizations.values()
            for mark_id in localization.mark_ids
            if mark_id in marks
        }
    )
    unassigned_mark_ids = sorted(set(marks) - set(assigned_mark_ids))
    if unassigned_mark_ids:
        reason_counts["unassigned_marks"] = len(unassigned_mark_ids)
    return {
        "reason_counts": reason_counts,
        "failed_items": failed_items,
        "unassigned_mark_ids": unassigned_mark_ids,
        "quality": (passed_count, len(assigned_mark_ids)),
    }


def recognize_question_batch(
    client,
    image_path: str,
    subject_hint: Optional[str],
    confidence_threshold: float,
    mark_confidence_threshold: float,
    localization_threshold: float,
    localization_max_area_ratio: float,
    red_pixel_min_ratio: float,
    red_pixel_expansion_ratio: float,
    tag_config_path: str,
    ocr_verifier,
    crop_context_padding_ratio: float = 0.0,
    recognition_correction: Optional[str] = None,
    local_red_scan: Optional[RedMarkScanResult] = None,
    mark_mismatch_retry_count: int = 0,
    ocr_full_page_max_edge: int = 1600,
    ocr_crop_recheck_limit: int = 3,
    force_mode: Optional[Literal["marked", "unmarked"]] = None,
    correction_group_enabled: bool = True,
    pair_max_distance_ratio: float = 0.12,
    dedup_iou_threshold: float = 0.8,
    anchor_max_gap_ratio: float = 0.0,
    cross_only_max_gap_ratio: float = 0.0,
    semantic_retry_count: int = 0,
    marked_ocr_recheck_limit: int = 0,
) -> tuple[VisionResult, List[dict]]:
    """Recognize, localize, and apply adaptive local evidence policy."""
    legacy_mode = local_red_scan is None
    scan_detected = bool(local_red_scan and local_red_scan.status == "detected")
    mode = force_mode or ("marked" if scan_detected or legacy_mode else "unmarked")
    local_red_regions = [
        list(region.bbox) for region in (local_red_scan.regions if local_red_scan else [])
    ]

    correction_group_validation = {
        "raw_mark_count": 0,
        "correction_group_count": 0,
        "paired_group_count": 0,
        "single_mark_group_count": 0,
        "deduplicated_mark_count": 0,
    }
    for attempt in range(mark_mismatch_retry_count + 1):
        result = _recognize_with_mode(
            client,
            image_path,
            subject_hint,
            recognition_correction,
            mode,
            local_red_regions,
        )
        try:
            valid_marks, rejected_mark_ids, mark_diagnostics = filter_valid_error_marks(
                image_path,
                result.error_marks,
                confidence_threshold=mark_confidence_threshold,
                red_pixel_min_ratio=red_pixel_min_ratio,
                expansion_ratio=red_pixel_expansion_ratio,
            )
        except ErrorMarkImageInvalid:
            valid_marks = []
            rejected_mark_ids = [mark.mark_id for mark in result.error_marks]
            mark_diagnostics = []
        if correction_group_enabled:
            valid_marks, correction_group_validation = normalize_error_mark_groups(
                valid_marks,
                dedup_iou_threshold=dedup_iou_threshold,
                pair_max_distance_ratio=pair_max_distance_ratio,
            )
        else:
            correction_group_validation = {
                "raw_mark_count": len(valid_marks),
                "correction_group_count": len(valid_marks),
                "paired_group_count": sum(
                    mark.mark_type == "cross_circle" for mark in valid_marks
                ),
                "single_mark_group_count": sum(
                    mark.mark_type != "cross_circle" for mark in valid_marks
                ),
                "deduplicated_mark_count": 0,
            }
        if mode != "marked" or valid_marks or not scan_detected:
            break
        if attempt == mark_mismatch_retry_count:
            raise ImageReviewRequired(
                "red_marks_unresolved",
                "系统检测到批改痕迹，但没有找到可确认的错题区域。",
                diagnostic={
                    "operation": "recognition",
                    "reason": "local_red_marks_without_model_marks",
                    "mark_count": len(result.error_marks),
                },
            )

    if mode == "unmarked" and valid_marks and force_mode != "unmarked":
        mode = "marked"
    elif legacy_mode and not valid_marks:
        mode = "unmarked"
    marks_by_id = {mark.mark_id: mark for mark in valid_marks}

    localizations = {}
    assignment_sources: dict[int, str] = {}
    localization_batch_validation = {
        "status": "skipped",
        "error_code": None,
        "error_reason": None,
        "error_diagnostic": {},
        "returned_count": 0,
        "validated_count": 0,
        "verified_count": 0,
        "reliable_mark_count": 0,
        "rejection_counts": {},
        "geometry_failure_counts": {},
        "geometry_diagnostics": [],
        "assigned_mark_ids": [],
        "unassigned_mark_ids": [],
        "unassigned_mark_count": 0,
        "missing_mark_diagnostics": [],
        "assignment_diagnostics": [],
        "semantic_retry_attempts": 0,
        "semantic_retry_reason_counts": {},
        "marked_ocr_recheck_count": 0,
    }
    try:
        if (
            (mode == "unmarked" and not legacy_mode)
            or not result.error_marks
            or valid_marks
        ):
            localization_result = _localize_with_correction(
                client,
                image_path,
                result.items,
                valid_marks,
            )
            localization_batch_validation["returned_count"] = len(
                localization_result.items
            )
            localizations = validated_localizations(
                localization_result,
                item_count=len(result.items),
                marks=marks_by_id,
            )
            assignment_sources = {
                index: "model"
                for index, localization in localizations.items()
                if localization.mark_ids
            }
            if correction_group_enabled:
                localizations, assignment_diagnostics = (
                    repair_unique_mark_assignments(
                        localizations,
                        marks_by_id,
                        result.items,
                        localization_threshold=localization_threshold,
                        max_area_ratio=localization_max_area_ratio,
                        anchor_max_gap_ratio=anchor_max_gap_ratio,
                        cross_only_max_gap_ratio=cross_only_max_gap_ratio,
                    )
                )
                localization_batch_validation["assignment_diagnostics"] = (
                    assignment_diagnostics
                )
                assignment_sources.update(
                    {
                        diagnostic["index"]: "deterministic"
                        for diagnostic in assignment_diagnostics
                    }
                )
            semantic_diagnostic = localization_semantic_diagnostic(
                localizations,
                marks_by_id,
                max_area_ratio=localization_max_area_ratio,
                anchor_max_gap_ratio=anchor_max_gap_ratio,
                cross_only_max_gap_ratio=cross_only_max_gap_ratio,
            )
            localization_batch_validation["semantic_retry_reason_counts"] = dict(
                semantic_diagnostic["reason_counts"]
            )
            for _retry_index in range(
                semantic_retry_count if correction_group_enabled else 0
            ):
                if not semantic_diagnostic["reason_counts"]:
                    break
                correction = {
                    "reason_counts": semantic_diagnostic["reason_counts"],
                    "failed_items": semantic_diagnostic["failed_items"],
                    "unassigned_mark_ids": semantic_diagnostic[
                        "unassigned_mark_ids"
                    ],
                    "correction_events": [
                        {
                            "mark_id": mark.mark_id,
                            "mark_type": mark.mark_type,
                            "bbox": mark.bbox,
                            "cross_bbox": mark.cross_bbox,
                            "circle_bbox": mark.circle_bbox,
                        }
                        for mark in valid_marks
                    ],
                }
                localization_batch_validation["semantic_retry_attempts"] += 1
                try:
                    retry_result = _localize_with_correction(
                        client,
                        image_path,
                        result.items,
                        valid_marks,
                        correction,
                    )
                    retry_localizations = validated_localizations(
                        retry_result,
                        item_count=len(result.items),
                        marks=marks_by_id,
                    )
                    retry_localizations, _retry_assignments = (
                        repair_unique_mark_assignments(
                            retry_localizations,
                            marks_by_id,
                            result.items,
                            localization_threshold=localization_threshold,
                            max_area_ratio=localization_max_area_ratio,
                            anchor_max_gap_ratio=anchor_max_gap_ratio,
                            cross_only_max_gap_ratio=cross_only_max_gap_ratio,
                        )
                    )
                    retry_diagnostic = localization_semantic_diagnostic(
                        retry_localizations,
                        marks_by_id,
                        max_area_ratio=localization_max_area_ratio,
                        anchor_max_gap_ratio=anchor_max_gap_ratio,
                        cross_only_max_gap_ratio=cross_only_max_gap_ratio,
                    )
                except VisionRecognitionError:
                    break
                if retry_diagnostic["quality"] > semantic_diagnostic["quality"]:
                    localizations = retry_localizations
                    semantic_diagnostic = retry_diagnostic
                    assignment_sources = {
                        index: "semantic_retry"
                        for index, localization in localizations.items()
                        if localization.mark_ids
                    }
                    localization_batch_validation["returned_count"] = len(
                        retry_result.items
                    )
                    localization_batch_validation["assignment_diagnostics"] = [
                        {
                            "index": index,
                            "mark_ids": list(localization.mark_ids),
                            "assignment_source": "semantic_retry",
                        }
                        for index, localization in sorted(localizations.items())
                        if localization.mark_ids
                    ]
            localization_batch_validation["status"] = "validated"
            localization_batch_validation["validated_count"] = len(localizations)
            assigned_mark_ids = sorted(
                {
                    mark_id
                    for localization in localizations.values()
                    for mark_id in localization.mark_ids
                }
            )
            unassigned_mark_ids = sorted(set(marks_by_id) - set(assigned_mark_ids))
            localization_batch_validation["assigned_mark_ids"] = assigned_mark_ids
            localization_batch_validation["unassigned_mark_ids"] = (
                unassigned_mark_ids
            )
            localization_batch_validation["unassigned_mark_count"] = len(
                unassigned_mark_ids
            )
    except VisionRecognitionError as exc:
        localization_batch_validation["status"] = "rejected"
        localization_batch_validation["error_code"] = exc.code
        localization_batch_validation["error_reason"] = exc.diagnostic.get("reason")
        localization_batch_validation["error_diagnostic"] = (
            safe_recognition_diagnostic(exc)
        )
        localizations = {}

    values = []
    for index, item in enumerate(result.items):
        localization = localizations.get(index)
        localization_red_validation = None
        localization_red_verified = False
        should_validate_local_red_evidence = (
            localization is not None
            and localization.matched
            and localization.bbox is not None
            and localization.confidence >= localization_threshold
            and bbox_area(localization.bbox) <= localization_max_area_ratio
            and localization_matches_evidence(localization, item)
            and bool(marks_by_id)
            and not localization.mark_ids
        )
        if should_validate_local_red_evidence:
            try:
                localization_red_validation = (
                    validate_localization_red_evidence(
                        image_path,
                        localization.bbox,
                        red_pixel_min_ratio=red_pixel_min_ratio,
                        expansion_ratio=red_pixel_expansion_ratio,
                    )
                )
                localization_red_verified = localization_red_validation[
                    "accepted"
                ]
            except ErrorMarkImageInvalid:
                localization_red_validation = {
                    "accepted": False,
                    "reason": "image_invalid",
                }
        question_values = build_question_values(
            item,
            index=index,
            confidence_threshold=confidence_threshold,
            localization=localization,
            localization_threshold=localization_threshold,
            localization_max_area_ratio=localization_max_area_ratio,
            marks=marks_by_id,
            normalized_tags=normalize_tags(
                item.tags,
                item.question_type,
                tag_config_path,
            ),
            crop_context_padding_ratio=crop_context_padding_ratio,
            localization_red_verified=localization_red_verified,
            anchor_max_gap_ratio=(
                anchor_max_gap_ratio if correction_group_enabled else 0.0
            ),
            cross_only_max_gap_ratio=(
                cross_only_max_gap_ratio if correction_group_enabled else 0.0
            ),
            circle_only_requires_review=correction_group_enabled,
        )
        question_values["ocr_raw_json"].update(
            {
                "error_marks": [
                    mark.model_dump(mode="json") for mark in result.error_marks
                ],
                "error_mark_validation": mark_diagnostics,
                "correction_group_validation": correction_group_validation,
                "valid_error_mark_ids": [mark.mark_id for mark in valid_marks],
                "rejected_error_mark_ids": rejected_mark_ids,
                "localization": (
                    localization.model_dump(mode="json")
                    if localization is not None
                    else None
                ),
                "localization_red_validation": localization_red_validation,
                "recognition_mode": mode,
                "assignment_source": assignment_sources.get(index),
            }
        )
        values.append(question_values)

    verifications = {}
    page_diagnostic = None
    ocr_enabled = bool(getattr(ocr_verifier, "enabled", True))
    if mode == "unmarked" and hasattr(ocr_verifier, "recognize_page"):
        from app.services.local_ocr_verification import classify_page_evidence

        page = ocr_verifier.recognize_page(image_path, ocr_full_page_max_edge)
        page_diagnostic = {
            "status": page.status,
            "duration_ms": page.duration_ms,
            "error_code": page.error_code,
            "prepared_size": page.prepared_size,
        }
        if page.status == "unavailable" and ocr_enabled:
            raise ImageReviewRequired(
                "local_ocr_unavailable",
                "本地文字校验暂不可用，请稍后重试或人工确认。",
                diagnostic={"operation": "local_ocr", "reason": page.error_code},
            )
        for index, question_values in enumerate(values):
            localization = localizations.get(index)
            if localization is not None and localization.bbox is not None:
                verifications[index] = classify_page_evidence(
                    page,
                    localization.bbox,
                    target_index=index,
                    items=result.items,
                    line_confidence_threshold=ocr_verifier.line_confidence_threshold,
                    min_effective_characters=ocr_verifier.min_effective_characters,
                    support_similarity_threshold=ocr_verifier.support_similarity_threshold,
                    contradiction_similarity_threshold=ocr_verifier.contradiction_similarity_threshold,
                )
        inconclusive_indexes = sorted(
            (
                index
                for index, verification in verifications.items()
                if verification.status == "inconclusive"
            ),
            key=lambda index: (-result.items[index].confidence, index),
        )[:ocr_crop_recheck_limit]
        for index in inconclusive_indexes:
            localization = localizations[index]
            verifications[index] = ocr_verifier.verify_crop(
                image_path,
                localization.bbox,
                target_index=index,
                items=result.items,
            )

    deterministic_rescue_indexes = {
        diagnostic["index"]
        for diagnostic in sorted(
            (
                diagnostic
                for diagnostic in localization_batch_validation[
                    "assignment_diagnostics"
                ]
                if diagnostic.get("assignment_source") == "deterministic"
            ),
            key=lambda diagnostic: (
                diagnostic.get("nearest_distance_ratio", 1.0),
                diagnostic["index"],
            ),
        )[:marked_ocr_recheck_limit]
    }
    for index, question_values in enumerate(values):
        reliable_mark = question_values["ocr_raw_json"]["reliable_error_mark"]
        localization = localizations.get(index)
        proposed_bbox = (
            localization.bbox
            if localization is not None
            and question_values["crop_region"].get("bbox") is not None
            else None
        )
        assignment_source = question_values["ocr_raw_json"]["assignment_source"]
        should_run_marked_ocr = assignment_source != "deterministic" or (
            index in deterministic_rescue_indexes
        )
        if (
            mode == "marked"
            and reliable_mark
            and proposed_bbox is not None
            and should_run_marked_ocr
        ):
            verify_crop = getattr(ocr_verifier, "verify_crop", ocr_verifier.verify)
            verifications[index] = verify_crop(
                image_path,
                proposed_bbox,
                target_index=index,
                items=result.items,
            )
            if assignment_source == "deterministic":
                localization_batch_validation["marked_ocr_recheck_count"] += 1
        elif mode == "unmarked" and not hasattr(ocr_verifier, "recognize_page") and proposed_bbox:
            verifications[index] = ocr_verifier.verify(
                image_path,
                proposed_bbox,
                target_index=index,
                items=result.items,
            )

        verification = verifications.get(index)
        local_ocr = (
            verification.model_dump(mode="json")
            if verification is not None
            else {
                "status": "disabled" if not ocr_enabled else "inconclusive",
                "matched_index": None,
                "text_summary": "",
                "confidence": None,
                "error_code": None,
                "duration_ms": 0.0,
            }
        )
        if local_ocr["status"] in {"wrong_candidate", "text_mismatch"}:
            question_values["crop_region"] = {
                "bbox_source": "unverified",
                "localization_status": "needs_review",
                "index": index,
            }
            question_values["review_status"] = "needs_review"
        question_values["ocr_raw_json"]["local_ocr"] = local_ocr
        question_values["ocr_raw_json"]["local_ocr_page"] = page_diagnostic
        decision = decide_candidate(
            mode=mode,
            student_answer=result.items[index].raw_text,
            reference_answer=result.items[index].answer,
            unanswered=not bool(result.items[index].raw_text.strip()),
            review_status=question_values["review_status"],
            reliable_error_mark=reliable_mark,
            local_ocr_status=local_ocr["status"],
            ocr_enabled=ocr_enabled,
        )
        question_values["collection_status"] = {
            "collect": "collected",
            "review": "pending_review",
            "discard": "ignored",
        }[decision.action]
        question_values["ocr_raw_json"]["collection_reason"] = decision.reason

    validation_keys = (
        "present",
        "matched",
        "has_bbox",
        "confidence_passed",
        "geometry_passed",
        "text_evidence_passed",
        "verified",
    )
    localization_batch_validation["verified_count"] = sum(
        bool(candidate["ocr_raw_json"]["localization_validation"]["verified"])
        for candidate in values
    )
    localization_batch_validation["reliable_mark_count"] = sum(
        bool(candidate["ocr_raw_json"]["reliable_error_mark"])
        for candidate in values
    )
    localization_batch_validation["rejection_counts"] = {
        key: sum(
            not bool(candidate["ocr_raw_json"]["localization_validation"][key])
            for candidate in values
        )
        for key in validation_keys
    }
    geometry_diagnostics = [
        {
            "index": index,
            **candidate["ocr_raw_json"]["localization_validation"]["geometry"],
        }
        for index, candidate in enumerate(values)
        if candidate["ocr_raw_json"]["localization_validation"]["geometry"]
        is not None
        and not candidate["ocr_raw_json"]["localization_validation"]["geometry"][
            "passed"
        ]
    ]
    geometry_failure_counts = {}
    for diagnostic in geometry_diagnostics:
        for failure_reason in diagnostic["failure_reasons"]:
            geometry_failure_counts[failure_reason] = (
                geometry_failure_counts.get(failure_reason, 0) + 1
            )
    localization_batch_validation["geometry_failure_counts"] = (
        geometry_failure_counts
    )
    localization_batch_validation["geometry_diagnostics"] = geometry_diagnostics
    missing_mark_diagnostics = []
    red_regions = local_red_scan.regions if local_red_scan is not None else []
    for index, candidate in enumerate(values):
        localization = localizations.get(index)
        if (
            localization is None
            or localization.bbox is None
            or localization.mark_ids
        ):
            continue
        missing_mark_diagnostics.append(
            {
                "index": index,
                "local_red_validation": safe_local_red_validation(
                    candidate["ocr_raw_json"].get("localization_red_validation")
                ),
                "nearest_local_red_region": nearest_red_region_diagnostic(
                    localization.bbox,
                    red_regions,
                ),
            }
        )
    localization_batch_validation["missing_mark_diagnostics"] = (
        missing_mark_diagnostics
    )
    for question_values in values:
        question_values["ocr_raw_json"]["localization_batch_validation"] = dict(
            localization_batch_validation
        )

    if (
        mode == "marked"
        and not legacy_mode
        and valid_marks
        and not any(
            candidate["collection_status"] in {"collected", "pending_review"}
            for candidate in values
        )
    ):
        localization_detail_diagnostic = {}
        if localization_batch_validation["geometry_diagnostics"]:
            localization_detail_diagnostic.update(
                {
                    "localization_geometry_failure_counts": (
                        localization_batch_validation["geometry_failure_counts"]
                    ),
                    "localization_geometry_diagnostics": (
                        localization_batch_validation["geometry_diagnostics"]
                    ),
                }
            )
        if localization_batch_validation["unassigned_mark_ids"]:
            localization_detail_diagnostic.update(
                {
                    "localization_assigned_mark_ids": (
                        localization_batch_validation["assigned_mark_ids"]
                    ),
                    "localization_unassigned_mark_ids": (
                        localization_batch_validation["unassigned_mark_ids"]
                    ),
                    "localization_unassigned_mark_count": (
                        localization_batch_validation["unassigned_mark_count"]
                    ),
                }
            )
        if localization_batch_validation["missing_mark_diagnostics"]:
            localization_detail_diagnostic["localization_missing_mark_diagnostics"] = (
                localization_batch_validation["missing_mark_diagnostics"]
            )
        raise ImageReviewRequired(
            "red_marks_unresolved",
            "系统检测到批改痕迹，但没有找到可确认的错题区域。",
            diagnostic={
                "operation": "localization",
                "reason": "marked_candidates_without_reliable_localization",
                "candidate_count": len(result.items),
                "mark_count": len(valid_marks),
                "localization_status": localization_batch_validation["status"],
                "localization_error_code": localization_batch_validation[
                    "error_code"
                ],
                "localization_error_reason": localization_batch_validation[
                    "error_reason"
                ],
                "localization_error_diagnostic": localization_batch_validation[
                    "error_diagnostic"
                ],
                "localization_returned_count": localization_batch_validation[
                    "returned_count"
                ],
                "localization_validated_count": localization_batch_validation[
                    "validated_count"
                ],
                "localization_verified_count": localization_batch_validation[
                    "verified_count"
                ],
                "localization_reliable_mark_count": localization_batch_validation[
                    "reliable_mark_count"
                ],
                "localization_rejection_counts": localization_batch_validation[
                    "rejection_counts"
                ],
                **localization_detail_diagnostic,
            },
        )
    return result, values


def image_status_for(question_values: List[dict]) -> str:
    """An image needs review when any recognized item needs review."""
    if any(
        values.get("collection_status") == "pending_review"
        or (
            "collection_status" not in values
            and values.get("review_status") == "needs_review"
        )
        for values in question_values
    ):
        return "needs_review"
    return "confirmed"


def prepare_image_data_url(
    image_path: str,
    max_edge: int,
    jpeg_quality: int,
    diagnostic: dict | None = None,
) -> str:
    """Normalize orientation and size, then return a JPEG data URL."""
    path = Path(image_path)
    if not path.is_file():
        raise VisionRecognitionError(
            "image_preprocessing_failed", "图片处理失败，请重新拍摄后提交"
        )

    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if diagnostic is not None:
                diagnostic.update(
                    {
                        "source_width": source.width,
                        "source_height": source.height,
                    }
                )
            if max(image.size) > max_edge:
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                image.thumbnail((max_edge, max_edge), resampling)
            if diagnostic is not None:
                diagnostic.update(
                    {
                        "prepared_width": image.width,
                        "prepared_height": image.height,
                    }
                )
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    except (OSError, ValueError) as exc:
        raise VisionRecognitionError(
            "image_preprocessing_failed", "图片处理失败，请重新拍摄后提交"
        ) from exc

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _json_boundary_character_type(character: str | None) -> str:
    if character is None:
        return "none"
    return {
        "{": "object_start",
        "}": "object_end",
        "[": "array_start",
        "]": "array_end",
        '"': "string_quote",
        ":": "colon",
        ",": "comma",
    }.get(character, "digit" if character.isdigit() else "other")


def _response_shape_diagnostic(content: str) -> dict:
    stripped = content.strip()
    output_match = OUTPUT_RE.fullmatch(stripped)
    candidate = output_match.group(1).strip() if output_match else stripped
    fence_match = FENCE_RE.fullmatch(candidate)
    candidate = fence_match.group(1).strip() if fence_match else candidate
    first_character = candidate[0] if candidate else None
    last_character = candidate[-1] if candidate else None
    return {
        "response_content_length": len(content),
        "has_markdown_fence": bool(fence_match),
        "has_output_wrapper": bool(output_match),
        "first_non_whitespace_char_type": _json_boundary_character_type(
            first_character
        ),
        "last_non_whitespace_char_type": _json_boundary_character_type(
            last_character
        ),
        "likely_truncated": bool(candidate and last_character not in {"}", "]"}),
    }


def _extract_json(content: str, diagnostic: dict | None = None) -> dict:
    if not isinstance(content, str) or not content.strip():
        raise VisionRecognitionError(
            "vision_response_empty",
            "识别结果为空，请稍后重试",
            {
                **(diagnostic or {}),
                "response_content_length": (
                    len(content) if isinstance(content, str) else 0
                ),
            },
        )

    shape_diagnostic = _response_shape_diagnostic(content)
    response_diagnostic = {
        **(diagnostic or {}),
        **shape_diagnostic,
    }
    if diagnostic is not None:
        diagnostic.update(shape_diagnostic)
    candidate = content.strip()
    output_match = OUTPUT_RE.fullmatch(candidate)
    if output_match:
        candidate = output_match.group(1).strip()
    fence_match = FENCE_RE.fullmatch(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise VisionRecognitionError(
            "vision_response_json_invalid",
            "识别结果格式异常，请稍后重试",
            {
                **response_diagnostic,
                "json_error_position": exc.pos,
                "json_error_line": exc.lineno,
                "json_error_column": exc.colno,
            },
        ) from exc
    if not isinstance(data, dict):
        raise VisionRecognitionError(
            "vision_response_json_invalid",
            "识别结果格式异常，请稍后重试",
            {
                **response_diagnostic,
                "parsed_json_type": type(data).__name__,
            },
        )
    return data


class MiniMaxVisionClient:
    def __init__(
        self,
        api_key: str,
        api_host: str,
        timeout_seconds: float,
        max_retries: int,
        max_edge: int,
        jpeg_quality: int,
        retry_delay_seconds: float = 1,
        transport=None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key
        self.api_host = api_host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_edge = max_edge
        self.jpeg_quality = jpeg_quality
        self.retry_delay_seconds = retry_delay_seconds
        self.transport = transport
        self.sleep = sleep

    @classmethod
    def from_settings(cls):
        return cls(
            api_key=settings.MINIMAX_API_KEY,
            api_host=settings.MINIMAX_API_HOST,
            timeout_seconds=settings.MINIMAX_VISION_TIMEOUT_SECONDS,
            max_retries=settings.MINIMAX_VISION_MAX_RETRIES,
            max_edge=settings.MINIMAX_IMAGE_MAX_EDGE,
            jpeg_quality=settings.MINIMAX_IMAGE_JPEG_QUALITY,
            retry_delay_seconds=settings.MINIMAX_VISION_RETRY_DELAY_SECONDS,
        )

    def recognize(
        self,
        image_path: str,
        subject_hint: Optional[str] = None,
        recognition_correction: Optional[str] = None,
        recognition_mode: Literal["marked", "unmarked"] = "marked",
        local_red_regions: Optional[List[List[float]]] = None,
    ) -> VisionResult:
        if not self.api_key or not self.api_host:
            raise VisionRecognitionError(
                "vision_not_configured", "识别服务尚未配置，请联系管理员"
            )

        diagnostic = {"operation": "recognition"}
        image_url = prepare_image_data_url(
            image_path,
            self.max_edge,
            self.jpeg_quality,
            diagnostic,
        )
        payload = {
            "prompt": recognition_prompt_for(
                recognition_mode,
                subject_hint,
                local_red_regions or [],
                recognition_correction,
            ),
            "image_url": image_url,
        }

        return self._request(payload, VisionResult, diagnostic)

    def localize(
        self,
        image_path: str,
        items: List[VisionItem],
        error_marks: List[ErrorMark],
        correction: Optional[dict] = None,
    ) -> LocalizationResult:
        diagnostic = {
            "operation": "localization",
            "candidate_count": len(items),
            "mark_count": len(error_marks),
        }
        image_url = prepare_image_data_url(
            image_path,
            self.max_edge,
            self.jpeg_quality,
            diagnostic,
        )
        item_summaries = [
            {
                "index": index,
                "instruction": item.instruction,
                "prompt_text": item.prompt_text,
                "raw_text": item.raw_text,
                "normalized_text": item.normalized_text,
                "answer": item.answer,
                "question_type": item.question_type,
            }
            for index, item in enumerate(items)
        ]
        prompt = LOCALIZATION_PROMPT.replace(
            "__INPUT__",
            json.dumps(
                {
                    "items": item_summaries,
                    "error_marks": [
                        mark.model_dump(mode="json") for mark in error_marks
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        if correction:
            prompt += (
                "\n\n定位纠偏：上一次定位存在下列结构或几何问题。"
                "请重新观察整张图片，修正标记归属与完整题目 bbox；"
                "仍须让每个 index 恰好返回一次，不得只返回失败子集。\n"
                + json.dumps(correction, ensure_ascii=False, separators=(",", ":"))
            )
        return self._request(
            {"prompt": prompt, "image_url": image_url},
            LocalizationResult,
            diagnostic,
        )

    def _request(self, payload, result_model, diagnostic: dict):
        request_payload = dict(payload)
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(request_payload)
                is_transient = response.status_code == 429 or 500 <= response.status_code < 600
                if is_transient and attempt < self.max_retries:
                    self.sleep(self.retry_delay_seconds)
                    continue
                if response.status_code < 200 or response.status_code >= 300:
                    if response.status_code == 429:
                        raise VisionRecognitionError(
                            "vision_rate_limited", "识别服务繁忙，请稍后重试", diagnostic
                        )
                    if 500 <= response.status_code < 600:
                        raise VisionRecognitionError(
                            "vision_service_unavailable", "识别服务暂时不可用，请稍后重试", diagnostic
                        )
                    raise VisionRecognitionError(
                        "vision_http_rejected",
                        "识别服务返回异常，请稍后重试",
                        {**diagnostic, "status_code": response.status_code},
                    )
                raw = self._parse_response_content(response, diagnostic)
                try:
                    return result_model.model_validate(raw)
                except ValidationError as exc:
                    raise VisionRecognitionError(
                        "vision_response_schema_invalid",
                        "识别结果格式不完整，请稍后重试",
                        {
                            **diagnostic,
                            "validation_fields": [
                                ".".join(str(part) for part in error["loc"])
                                for error in exc.errors()[:5]
                            ],
                        },
                    ) from exc
            except VisionRecognitionError as exc:
                exc.diagnostic = {
                    **exc.diagnostic,
                    "response_attempt": attempt + 1,
                    "response_max_attempts": self.max_retries + 1,
                }
                if (
                    exc.code in FORMAT_RETRYABLE_ERROR_CODES
                    and attempt < self.max_retries
                ):
                    logger.info(
                        "vision_response_retry operation=%s error_code=%s "
                        "attempt=%s max_retries=%s diagnostic=%s",
                        diagnostic.get("operation", "unknown"),
                        exc.code,
                        attempt + 1,
                        self.max_retries,
                        json.dumps(
                            safe_recognition_diagnostic(exc),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                    prompt = request_payload.get("prompt")
                    if (
                        isinstance(prompt, str)
                        and FORMAT_RETRY_INSTRUCTION not in prompt
                    ):
                        request_payload = {
                            **request_payload,
                            "prompt": prompt + FORMAT_RETRY_INSTRUCTION,
                        }
                    self.sleep(self.retry_delay_seconds)
                    continue
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < self.max_retries:
                    self.sleep(self.retry_delay_seconds)
                    continue
                code, message = (
                    ("vision_timeout", "识别服务响应超时，请稍后重试")
                    if isinstance(exc, httpx.TimeoutException)
                    else ("vision_service_unavailable", "识别服务暂时不可用，请稍后重试")
                )
                raise VisionRecognitionError(code, message, diagnostic) from exc

        raise VisionRecognitionError(
            "vision_service_unavailable", "识别服务暂时不可用，请稍后重试"
        )

    def _post(self, payload):
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "MM-API-Source": "Minimax-MCP",
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            return client.post(self.api_host + VISION_PATH, headers=headers, json=payload)

    @staticmethod
    def _parse_response_content(response: httpx.Response, diagnostic: dict) -> dict:
        try:
            data = response.json()
        except ValueError as exc:
            raise VisionRecognitionError(
                "vision_response_envelope_invalid",
                "识别服务返回格式异常，请稍后重试",
                diagnostic,
            ) from exc
        if not isinstance(data, dict):
            raise VisionRecognitionError(
                "vision_response_envelope_invalid",
                "识别服务返回格式异常，请稍后重试",
                diagnostic,
            )

        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0):
            raise VisionRecognitionError(
                "vision_upstream_rejected",
                "识别服务返回异常，请稍后重试",
                {**diagnostic, "provider_status_code": status_code},
            )

        return _extract_json(data.get("content", ""), diagnostic)
