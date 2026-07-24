"""MiniMax Token Plan image understanding and structured recognition."""

import base64
import io
import json
import re
import time
import unicodedata
from pathlib import Path
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
    filter_valid_error_marks,
)
from app.services.tag_normalization import normalize_tags


VISION_PATH = "/v1/coding_plan/vlm"
OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

RECOGNITION_PROMPT = """你是小学错题内容与红色批改标记识别器。请观察整张图片，一次性输出题目内容和独立的红色错误标记。

要求：
1. 忽略 Date、日期栏、页码、装订线、空白横线和与错题无关的印刷页眉。
2. 识别老师用于标识错误的红圈、红叉、红色删除线、红色波浪线、红色下划线或红色纠错批注；不要把印刷装饰色或单独的红色对勾误判为错误标记。
3. 同一作答单元上的红圈、红叉和纠正笔迹视为同一标记组。每个标记组只输出一个 error_mark，mark_id 从 0 开始连续且唯一。
4. error_mark.bbox 只覆盖实际可见的红色错误标记，使用归一化角点格式 [left, top, right, bottom]，满足 0 <= left < right <= 1 和 0 <= top < bottom <= 1。
5. 题目 item 与 error_marks 必须分别识别。题目 item 不得输出 bbox，也不得预先绑定 mark_id；后续步骤会根据整图内容独立完成匹配和定位。
6. item 的粒度必须是最小可独立作答单元，不能把整道编号大题合并成一个 item。拼音格、完整词语格组、单个填空、单个选择项或一道计算题分别视为独立作答单元。
7. 若存在明确的红色错误标记，每个被标记的独立作答单元输出一个 item；同一道编号大题中有多个被标记的小题时，必须分别输出多个 item。同一行存在多个兄弟小题时，不得加入未标记的兄弟小题。
8. 对于看词语写拼音、看拼音写词语等词语练习，最小可独立作答单元是完整词语格组。即使标记只覆盖单字、单音节或部分笔画，也要识别其所属完整词语；完整词语优先于红色标记的像素覆盖范围。
9. 完整词语格组的各字段必须保持同一范围：raw_text 抄录学生对整个词语的实际作答，prompt_text 填写整个印刷提示，answer 填写整个正确答案，question_type 根据完整提示与完整作答判断。
10. 所有字段必须从当前图片可见内容提取，不得复用提示中的示例或臆造图片中不存在的词语。
11. 若没有发现明确的红色错误标记，输出图片中的所有最小可独立作答单元，每个单元一个 item。
12. 红色批改符号本身不要写入 raw_text；老师写出的纠正内容可作为 answer 的参考。
13. raw_text 必须忠实抄录学生实际书写，包括错字、漏字、错误拼音和错误答案；禁止自动改正后覆盖原文。
14. instruction 必须填写图片中可见的原始练习要求；prompt_text 必须填写重新出卷时展示的干净提示材料。二者不得包含学生作答、正确答案或老师批改笔迹。
15. question_type 只能是 write_pinyin、write_word、fill_blank、calculation、other 之一。无法确认的内容保留原样并写入 uncertain_segments。
16. confidence 范围为 0 到 1。difficulty 必须是 1 到 5 的整数，1 表示很简单，5 表示很难。
17. tags 只能使用中文标签，例如“拼音”“词语”“错别字”“老师批改”，不得返回 pinyin、word、teacher-marked 等英文编码。
18. 只输出一个 JSON 对象，不要解释，不要 Markdown。

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
    "mark_type": "circle|cross|deletion|underline|annotation|mixed",
    "bbox": [0.0, 0.0, 1.0, 1.0],
    "confidence": 0.95
  }}],
  "ignored_text": ["被忽略的页眉"]
}}

科目提示：{subject_hint}
"""

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

    @field_validator("raw_text", "instruction", "prompt_text")
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
        "deletion",
        "underline",
        "annotation",
        "mixed",
    ]
    bbox: List[float]
    confidence: float = Field(ge=0, le=1)

    @field_validator("bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        return validate_normalized_bbox(value)


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
    indexes = [item.index for item in result.items]
    if len(indexes) != item_count or len(set(indexes)) != len(indexes):
        raise VisionRecognitionError("MiniMax localization indexes do not match recognition")
    if set(indexes) != set(range(item_count)):
        raise VisionRecognitionError("MiniMax localization indexes do not match recognition")
    assigned_mark_ids = [
        mark_id
        for item in result.items
        for mark_id in item.mark_ids
    ]
    if len(assigned_mark_ids) != len(set(assigned_mark_ids)):
        raise VisionRecognitionError("MiniMax localization assigned a mark more than once")
    if set(assigned_mark_ids) != set(marks):
        raise VisionRecognitionError(
            "MiniMax localization marks do not match validated error marks"
        )
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


def localization_passes_geometry(
    localization: LocalizationItem,
    marks: dict[int, ErrorMark],
    max_area_ratio: float,
) -> bool:
    if not localization.matched or localization.bbox is None:
        return False
    if bbox_area(localization.bbox) > max_area_ratio:
        return False
    if marks and not localization.mark_ids:
        return False
    return all(
        mark_id in marks
        and bbox_contains_center(localization.bbox, marks[mark_id].bbox)
        for mark_id in localization.mark_ids
    )


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


def build_question_values(
    item: VisionItem,
    index: int,
    confidence_threshold: float,
    localization: Optional[LocalizationItem],
    localization_threshold: float,
    localization_max_area_ratio: float,
    marks: dict[int, ErrorMark],
    normalized_tags: List[str],
) -> dict:
    """Map a validated vision item to the existing question persistence contract."""
    localization_verified = (
        localization is not None
        and localization.matched
        and localization.bbox is not None
        and localization.confidence >= localization_threshold
        and localization_passes_geometry(
            localization,
            marks=marks,
            max_area_ratio=localization_max_area_ratio,
        )
        and localization_matches_evidence(localization, item)
    )
    needs_review = (
        item.confidence < confidence_threshold
        or bool(item.uncertain_segments)
        or not localization_verified
    )
    crop_region = {
        "bbox_source": "unverified",
        "localization_status": "needs_review",
        "index": index,
    }
    if localization_verified:
        crop_region = {
            "bbox": localization.bbox,
            "bbox_format": "normalized_ltrb",
            "bbox_source": "minimax_marker_anchored",
            "bbox_confidence": localization.confidence,
            "localization_status": "verified",
            "mark_ids": localization.mark_ids,
            "index": index,
        }
    return {
        "crop_region": crop_region,
        "subject": item.subject,
        "ocr_text": item.raw_text,
        "ocr_answer": item.answer,
        "ocr_raw_json": {"provider": "minimax", **item.model_dump(mode="json")},
        "question_type": item.question_type,
        "tags": normalized_tags,
        "difficulty": item.difficulty,
        "status": "needs_review" if needs_review else "confirmed",
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
) -> tuple[VisionResult, List[dict]]:
    """Recognize marks/content, independently localize, then OCR-check each crop."""
    result = client.recognize(image_path, subject_hint=subject_hint)
    try:
        valid_marks, rejected_mark_ids = filter_valid_error_marks(
            image_path,
            result.error_marks,
            confidence_threshold=mark_confidence_threshold,
            red_pixel_min_ratio=red_pixel_min_ratio,
            expansion_ratio=red_pixel_expansion_ratio,
        )
    except ErrorMarkImageInvalid:
        valid_marks = []
        rejected_mark_ids = [mark.mark_id for mark in result.error_marks]
    marks_by_id = {mark.mark_id: mark for mark in valid_marks}

    localizations = {}
    try:
        if not result.error_marks or valid_marks:
            localizations = validated_localizations(
                client.localize(image_path, result.items, valid_marks),
                item_count=len(result.items),
                marks=marks_by_id,
            )
    except VisionRecognitionError:
        localizations = {}

    values = []
    for index, item in enumerate(result.items):
        localization = localizations.get(index)
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
        )
        local_ocr = {
            "status": "unavailable",
            "matched_index": None,
            "text_summary": "",
            "confidence": None,
        }
        if rejected_mark_ids:
            question_values["crop_region"] = {
                "bbox_source": "unverified",
                "localization_status": "needs_review",
                "index": index,
            }
            question_values["status"] = "needs_review"

        proposed_bbox = question_values["crop_region"].get("bbox")
        if proposed_bbox is not None:
            verification = ocr_verifier.verify(
                image_path,
                proposed_bbox,
                target_index=index,
                items=result.items,
            )
            local_ocr = verification.model_dump(mode="json")
            if verification.status == "contradict":
                question_values["crop_region"] = {
                    "bbox_source": "unverified",
                    "localization_status": "needs_review",
                    "index": index,
                }
                question_values["status"] = "needs_review"

        question_values["ocr_raw_json"].update(
            {
                "error_marks": [
                    mark.model_dump(mode="json") for mark in result.error_marks
                ],
                "valid_error_mark_ids": [mark.mark_id for mark in valid_marks],
                "rejected_error_mark_ids": rejected_mark_ids,
                "localization": (
                    localization.model_dump(mode="json")
                    if localization is not None
                    else None
                ),
                "local_ocr": local_ocr,
            }
        )
        values.append(question_values)
    return result, values


def image_status_for(question_values: List[dict]) -> str:
    """An image needs review when any recognized item needs review."""
    if any(values["status"] == "needs_review" for values in question_values):
        return "needs_review"
    return "confirmed"


def prepare_image_data_url(image_path: str, max_edge: int, jpeg_quality: int) -> str:
    """Normalize orientation and size, then return a JPEG data URL."""
    path = Path(image_path)
    if not path.is_file():
        raise VisionRecognitionError("Image file does not exist")

    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if max(image.size) > max_edge:
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                image.thumbnail((max_edge, max_edge), resampling)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    except (OSError, ValueError) as exc:
        raise VisionRecognitionError("Image preprocessing failed") from exc

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _extract_json(content: str) -> dict:
    if not isinstance(content, str) or not content.strip():
        raise VisionRecognitionError("MiniMax returned empty content")

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
        raise VisionRecognitionError("MiniMax returned invalid structured content") from exc
    if not isinstance(data, dict):
        raise VisionRecognitionError("MiniMax returned invalid structured content")
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

    def recognize(self, image_path: str, subject_hint: Optional[str] = None) -> VisionResult:
        if not self.api_key or not self.api_host:
            raise VisionRecognitionError("MiniMax vision is not configured")

        image_url = prepare_image_data_url(image_path, self.max_edge, self.jpeg_quality)
        payload = {
            "prompt": RECOGNITION_PROMPT.format(subject_hint=subject_hint or "自动判断"),
            "image_url": image_url,
        }

        return self._request(payload, VisionResult)

    def localize(
        self,
        image_path: str,
        items: List[VisionItem],
        error_marks: List[ErrorMark],
    ) -> LocalizationResult:
        image_url = prepare_image_data_url(image_path, self.max_edge, self.jpeg_quality)
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
        return self._request(
            {"prompt": prompt, "image_url": image_url},
            LocalizationResult,
        )

    def _request(self, payload, result_model):
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(payload)
                is_transient = response.status_code == 429 or 500 <= response.status_code < 600
                if is_transient and attempt < self.max_retries:
                    self.sleep(self.retry_delay_seconds)
                    continue
                if response.status_code < 200 or response.status_code >= 300:
                    raise VisionRecognitionError("MiniMax vision request failed with HTTP %s" % response.status_code)
                raw = self._parse_response_content(response)
                try:
                    return result_model.model_validate(raw)
                except ValidationError as exc:
                    raise VisionRecognitionError(
                        "MiniMax recognition result failed validation"
                    ) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < self.max_retries:
                    self.sleep(self.retry_delay_seconds)
                    continue
                raise VisionRecognitionError("MiniMax vision request failed") from exc

        raise VisionRecognitionError("MiniMax vision request failed")

    def _post(self, payload):
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "MM-API-Source": "Minimax-MCP",
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            return client.post(self.api_host + VISION_PATH, headers=headers, json=payload)

    @staticmethod
    def _parse_response_content(response: httpx.Response) -> dict:
        try:
            data = response.json()
        except ValueError as exc:
            raise VisionRecognitionError("MiniMax returned an invalid response") from exc
        if not isinstance(data, dict):
            raise VisionRecognitionError("MiniMax returned an invalid response")

        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0):
            raise VisionRecognitionError("MiniMax vision API rejected the request with code %s" % status_code)

        return _extract_json(data.get("content", ""))
