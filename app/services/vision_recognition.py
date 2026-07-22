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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config import settings


VISION_PATH = "/v1/coding_plan/vlm"
OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

RECOGNITION_PROMPT = """你是小学错题图片识别器。请直接观察图片，一次性完成文字识别、版面分组和结构化。

要求：
1. 忽略 Date、日期栏、页码、装订线、空白横线和与错题无关的印刷页眉。
2. 先判断是否存在老师用于标识错误的红圈、红叉、红色删除线、红色波浪线或红色纠错批注；不要把印刷装饰色或单独的红色对勾误判为错误标记。
3. item 的粒度必须是最小可独立作答单元，不能把整道编号大题合并成一个 item。拼音格、词语格、单个填空、单个选择项或一道计算题分别视为独立作答单元。
4. 若存在明确的红色错误标记，每个被标记的独立作答单元输出一个 item；同一道编号大题中有多个被标记的小题时，必须分别输出多个 item。只保留该小题自身的提示、学生作答和理解答案所需的局部上下文，禁止加入同一行或同一大题内未标记的兄弟小题。
5. 同一作答单元上的红圈、红叉和纠正笔迹视为同一标记组，不得重复拆成多条。若只圈出一个字或一个拼音音节，应保留它所属的完整词语或格组，但不能扩展到相邻词语。
6. 示例：一行包含“识字、组词、计算、图形、合唱”，只有“计算”的拼音被红圈标错时，raw_text 只包含“计算”及其学生拼音，不得包含“识字、组词、图形、合唱”。
7. 若没有发现明确的红色错误标记，输出图片中的所有最小可独立作答单元，每个单元一个 item，仍然不能按整道编号大题合并。
8. 红色批改符号本身不要写入 raw_text，老师写出的纠正内容可作为 answer 的参考。
9. bbox 必须使用归一化角点格式 [left, top, right, bottom]，满足 0 <= left < right <= 1 和 0 <= top < bottom <= 1，并只覆盖该独立作答单元及必要局部提示。
10. raw_text 必须忠实抄录该单元中学生实际书写，包括错字、漏字、错误拼音和错误答案；禁止自动改正后覆盖原文。
11. normalized_text 可填写该单元的规范写法；answer 可填写正确答案。无法确认时保留原样并写入 uncertain_segments。
12. confidence 是对 raw_text、红色标记关联和分组正确性的综合置信度，范围 0 到 1。
13. 只输出一个 JSON 对象，不要解释，不要 Markdown。

JSON 格式：
{{
  "items": [{{
    "raw_text": "学生实际书写",
    "normalized_text": "规范内容或 null",
    "answer": "正确答案或 null",
    "subject": "math|chinese|english",
    "question_type": "题型或 null",
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


class VisionRecognitionError(RuntimeError):
    """Safe recognition error that never contains credentials or image bytes."""


class VisionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    raw_text: str
    normalized_text: Optional[str] = None
    answer: Optional[str] = None
    subject: Literal["math", "chinese", "english"]
    question_type: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    difficulty: int = Field(default=3, ge=1, le=5)
    confidence: float = Field(ge=0, le=1)
    uncertain_segments: List[str] = Field(default_factory=list)
    bbox: Optional[List[float]] = None

    @field_validator("raw_text")
    @classmethod
    def raw_text_must_not_be_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("raw_text must not be blank")
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


def build_question_values(item: VisionItem, index: int, confidence_threshold: float) -> dict:
    """Map a validated vision item to the existing question persistence contract."""
    needs_review = item.confidence < confidence_threshold or bool(item.uncertain_segments)
    return {
        "crop_region": {
            "bbox": item.bbox,
            "bbox_format": "normalized_ltrb",
            "index": index,
        },
        "subject": item.subject,
        "ocr_text": item.raw_text,
        "ocr_answer": item.answer,
        "ocr_raw_json": {"provider": "minimax", **item.model_dump(mode="json")},
        "question_type": item.question_type,
        "tags": item.tags,
        "difficulty": item.difficulty,
        "status": "needs_review" if needs_review else "confirmed",
    }


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

        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(payload)
                is_transient = response.status_code == 429 or 500 <= response.status_code < 600
                if is_transient and attempt < self.max_retries:
                    self.sleep(self.retry_delay_seconds)
                    continue
                if response.status_code < 200 or response.status_code >= 300:
                    raise VisionRecognitionError("MiniMax vision request failed with HTTP %s" % response.status_code)
                result = self._parse_response(response)
                return result
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
    def _parse_response(response: httpx.Response) -> VisionResult:
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

        raw = _extract_json(data.get("content", ""))
        try:
            return VisionResult.model_validate(raw)
        except ValidationError as exc:
            raise VisionRecognitionError("MiniMax recognition result failed validation") from exc
