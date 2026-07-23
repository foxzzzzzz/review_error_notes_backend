"""MiniMax Token Plan image understanding and structured recognition."""

import base64
import io
import json
import re
import time
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
from app.services.tag_normalization import normalize_tags


VISION_PATH = "/v1/coding_plan/vlm"
OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

RECOGNITION_PROMPT = """你是小学错题图片识别器。请直接观察图片，一次性完成文字识别、版面分组和结构化。

要求：
1. 忽略 Date、日期栏、页码、装订线、空白横线和与错题无关的印刷页眉。
2. 先判断是否存在老师用于标识错误的红圈、红叉、红色删除线、红色波浪线或红色纠错批注；不要把印刷装饰色或单独的红色对勾误判为错误标记。
3. item 的粒度必须是最小可独立作答单元，不能把整道编号大题合并成一个 item。拼音格、完整词语格组、单个填空、单个选择项或一道计算题分别视为独立作答单元。对于看词语写拼音、看拼音写词语等词语类练习，最小可独立作答单元是一个完整词语格组，而不是其中单独一个汉字或拼音音节。
4. 若存在明确的红色错误标记，每个被标记的独立作答单元输出一个 item；同一道编号大题中有多个被标记的小题时，必须分别输出多个 item。只保留该小题自身的提示、学生作答和理解答案所需的局部上下文，禁止加入同一行或同一大题内未标记的兄弟小题。
5. 同一作答单元上的红圈、红叉和纠正笔迹视为同一标记组，不得重复拆成多条。若红圈或红叉只覆盖一个汉字、一个拼音音节或部分笔画，必须向该格组左右扩展并识别它所属的完整词语；完整词语优先于红色标记的像素覆盖范围，但不能扩展到相邻词语或未标记的兄弟小题。
6. 完整词语格组的各字段必须保持同一范围：raw_text 抄录学生对整个词语的实际作答，prompt_text 填写整个印刷提示词语或拼音，answer 填写整个正确答案，question_type 根据完整提示与完整作答之间的关系判断，bbox 覆盖该完整印刷提示与作答格组。不得把这些字段缩成被红色标记直接覆盖的单字或单音节。
7. 判断完整词语格组时，必须从当前图片读取印刷提示、学生实际作答和老师批改，不得复用或臆造本提示中的任何说明文字。
8. 即使红色标记只覆盖一个字、一个音节或部分笔画，也必须沿同一格组扩展到完整作答单元；所有字段都从该单元的可见内容提取，不得补入图片中不存在的词语。
9. 同一行存在多个兄弟小题时，只输出带明确错误标记的小题；相邻但未标记的小题不得进入该 item 的字段或 bbox。
10. 若没有发现明确的红色错误标记，输出图片中的所有最小可独立作答单元，每个单元一个 item，仍然不能按整道编号大题合并。
11. 红色批改符号本身不要写入 raw_text，老师写出的纠正内容可作为 answer 的参考。
12. bbox 必须使用归一化角点格式 [left, top, right, bottom]，满足 0 <= left < right <= 1 和 0 <= top < bottom <= 1，并只覆盖该独立作答单元及必要局部提示。
13. raw_text 必须忠实抄录该单元中学生实际书写，包括错字、漏字、错误拼音和错误答案；禁止自动改正后覆盖原文。
14. instruction 必须填写该独立作答单元在图片中可见的原始练习要求；prompt_text 必须填写重新出卷时展示的干净提示材料。二者都必须从当前图片提取，不得包含学生作答、正确答案、老师批改笔迹或提示词中的说明文字。
15. question_type 只能是 write_pinyin、write_word、fill_blank、calculation、other 之一。normalized_text 可填写该单元的规范写法；answer 可填写正确答案。无法确认时保留原样并写入 uncertain_segments。
16. confidence 是对 raw_text、红色标记关联和分组正确性的综合置信度，范围 0 到 1。
17. difficulty 必须是 1 到 5 的整数，1 表示很简单，5 表示很难；不得返回 0、负数或大于 5 的值。
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
    "tags": ["标签"],
    "difficulty": 1,
    "confidence": 0.95,
    "uncertain_segments": [],
    "bbox": [0.0, 0.0, 1.0, 1.0]
  }}],
  "ignored_text": ["被忽略的页眉"]
}}

科目提示：{subject_hint}
"""

LOCALIZATION_PROMPT = """你是小学错题区域定位器。请根据图片和已识别题目，只复核每道题的完整独立作答区域。

要求：
1. 输入中的 recognition_bbox 是第一次识别的候选 bbox。必须先检查该候选区域及其邻近范围是否真实出现对应的印刷提示、学生答案或相关红色批改标记。
2. 找不到对应可见内容、内容与输入题目不一致或无法确认时，必须返回 matched=false、bbox=null；禁止猜测位置。
3. matched=true 时，bbox 必须覆盖该题的印刷提示、学生答案和相关红色批改标记，并与第一次识别的候选 bbox 保持重叠；只允许为补全同一作答单元而扩展或收缩，禁止跳到其他区域。
4. 不得包含未标记的相邻兄弟小题，也不得把一个题目的 bbox 配给另一个题目。
5. bbox 使用归一化角点格式 [left, top, right, bottom]。
6. 对下面每个 index 恰好返回一次，不得缺失、重复或新增 index。
7. confidence 表示题目内容匹配和完整区域定位的综合置信度，范围 0 到 1。
8. 只输出 JSON，不要解释，不要 Markdown。

待定位题目：
__ITEMS__

返回格式：
{"items":[{"index":0,"matched":true,"bbox":[0.0,0.0,1.0,1.0],"confidence":0.95}]}
"""


class VisionRecognitionError(RuntimeError):
    """Safe recognition error that never contains credentials or image bytes."""


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
    bbox: Optional[List[float]] = None

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

    @field_validator("bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        if value is None:
            return value
        if len(value) != 4 or any(coordinate < 0 or coordinate > 1 for coordinate in value):
            raise ValueError("bbox must contain four normalized coordinates")
        left, top, right, bottom = value
        if left >= right or top >= bottom:
            raise ValueError("bbox must contain ordered left, top, right, bottom coordinates")
        return value


class VisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: List[VisionItem] = Field(min_length=1)
    ignored_text: List[str] = Field(default_factory=list)


class LocalizationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int = Field(ge=0)
    matched: bool
    bbox: Optional[List[float]] = None
    confidence: float = Field(ge=0, le=1)

    @field_validator("bbox")
    @classmethod
    def bbox_must_be_normalized(cls, value):
        if value is None:
            return value
        if len(value) != 4 or any(coordinate < 0 or coordinate > 1 for coordinate in value):
            raise ValueError("bbox must contain four normalized coordinates")
        left, top, right, bottom = value
        if left >= right or top >= bottom:
            raise ValueError("bbox must contain ordered left, top, right, bottom coordinates")
        return value

    @model_validator(mode="after")
    def matched_result_must_have_bbox(self):
        if self.matched != (self.bbox is not None):
            raise ValueError("matched localization must have bbox and unmatched must not")
        return self


class LocalizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: List[LocalizationItem] = Field(min_length=1)


def validated_localizations(
    result: LocalizationResult,
    item_count: int,
) -> dict[int, LocalizationItem]:
    indexes = [item.index for item in result.items]
    if len(indexes) != item_count or len(set(indexes)) != len(indexes):
        raise VisionRecognitionError("MiniMax localization indexes do not match recognition")
    if set(indexes) != set(range(item_count)):
        raise VisionRecognitionError("MiniMax localization indexes do not match recognition")
    return {item.index: item for item in result.items}


def normalized_bbox_iou(first: List[float], second: List[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def bbox_contains_center(container: List[float], candidate: List[float]) -> bool:
    center_x = (candidate[0] + candidate[2]) / 2
    center_y = (candidate[1] + candidate[3]) / 2
    return (
        container[0] <= center_x <= container[2]
        and container[1] <= center_y <= container[3]
    )


def build_question_values(
    item: VisionItem,
    index: int,
    confidence_threshold: float,
    localization: Optional[LocalizationItem],
    localization_threshold: float,
    localization_min_iou: float,
    normalized_tags: List[str],
) -> dict:
    """Map a validated vision item to the existing question persistence contract."""
    localization_verified = (
        localization is not None
        and localization.matched
        and localization.bbox is not None
        and item.bbox is not None
        and localization.confidence >= localization_threshold
        and normalized_bbox_iou(item.bbox, localization.bbox) >= localization_min_iou
        and bbox_contains_center(localization.bbox, item.bbox)
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
            "bbox_source": "minimax_verified",
            "bbox_confidence": localization.confidence,
            "localization_status": "verified",
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
    localization_threshold: float,
    localization_min_iou: float,
    tag_config_path: str,
) -> tuple[VisionResult, List[dict]]:
    """Recognize content, then verify all item regions in one localization call."""
    result = client.recognize(image_path, subject_hint=subject_hint)
    try:
        localizations = validated_localizations(
            client.localize(image_path, result.items),
            item_count=len(result.items),
        )
    except VisionRecognitionError:
        localizations = {}

    values = [
        build_question_values(
            item,
            index=index,
            confidence_threshold=confidence_threshold,
            localization=localizations.get(index),
            localization_threshold=localization_threshold,
            localization_min_iou=localization_min_iou,
            normalized_tags=normalize_tags(
                item.tags,
                item.question_type,
                tag_config_path,
            ),
        )
        for index, item in enumerate(result.items)
    ]
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
    ) -> LocalizationResult:
        image_url = prepare_image_data_url(image_path, self.max_edge, self.jpeg_quality)
        item_summaries = [
            {
                "index": index,
                "prompt_text": item.prompt_text,
                "raw_text": item.raw_text,
                "question_type": item.question_type,
                "recognition_bbox": item.bbox,
            }
            for index, item in enumerate(items)
        ]
        prompt = LOCALIZATION_PROMPT.replace(
            "__ITEMS__",
            json.dumps(item_summaries, ensure_ascii=False, indent=2),
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
