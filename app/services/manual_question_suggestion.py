"""OCR and DeepSeek suggestions for manually selected question regions."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.services.deepseek_vision import DeepSeekVisionClient
from app.services.local_ocr_verification import RapidOCRVerifier
from app.services.question_image import load_cropped_rgb_image


class ManualSuggestionFields(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    instruction: str = Field(max_length=500)
    prompt_text: str = Field(max_length=5000)
    question_type: Literal[
        "", "write_pinyin", "write_word", "fill_blank", "calculation", "other"
    ]
    correct_answer: str = Field(max_length=1000)
    student_answer: str = Field(max_length=1000)


def _new_ocr_verifier() -> RapidOCRVerifier:
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


def _context_bbox(bbox: list[float]) -> list[float]:
    left, top, right, bottom = bbox
    padding = settings.QUESTION_CROP_CONTEXT_PADDING_RATIO
    width = right - left
    height = bottom - top
    return [
        max(0.0, left - width * padding),
        max(0.0, top - height * padding),
        min(1.0, right + width * padding),
        min(1.0, bottom + height * padding),
    ]


def _image_data_url(
    image_path: str,
    context_bbox: list[float],
    selected_bbox: list[float],
) -> str:
    crop = load_cropped_rgb_image(
        image_path,
        context_bbox,
        max_pixels=settings.QUESTION_IMAGE_MAX_PIXELS,
    )
    left, top, right, bottom = context_bbox
    crop_width, crop_height = crop.size
    selection = [
        round((selected_bbox[0] - left) / (right - left) * crop_width),
        round((selected_bbox[1] - top) / (bottom - top) * crop_height),
        round((selected_bbox[2] - left) / (right - left) * crop_width),
        round((selected_bbox[3] - top) / (bottom - top) * crop_height),
    ]
    line_width = max(2, min(crop.size) // 500)
    ImageDraw.Draw(crop).rectangle(selection, outline=(0, 80, 255), width=line_width)
    if max(crop.size) > settings.MINIMAX_IMAGE_MAX_EDGE:
        crop.thumbnail(
            (settings.MINIMAX_IMAGE_MAX_EDGE, settings.MINIMAX_IMAGE_MAX_EDGE),
            Image.Resampling.LANCZOS,
        )
    output = BytesIO()
    crop.save(output, format="JPEG", quality=settings.MINIMAX_IMAGE_JPEG_QUALITY)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _ocr_text(image_path: str, bbox: list[float]) -> str:
    evidence = _new_ocr_verifier().recognize_crop(image_path, bbox)
    if evidence.status != "available":
        return ""
    return "\n".join(line.text.strip() for line in evidence.lines if line.text.strip())


def _llm_fields(
    image_path: str,
    bbox: list[float],
    *,
    subject: str | None,
    grade: int,
    semester: int,
) -> ManualSuggestionFields:
    context_bbox = _context_bbox(bbox)
    image_url = _image_data_url(image_path, context_bbox, bbox)
    prompt = Path(settings.MANUAL_QUESTION_SUGGESTION_PROMPT_PATH).read_text(
        encoding="utf-8"
    )
    prompt = (
        prompt.replace("{subject}", subject or "unknown")
        .replace("{grade}", str(grade))
        .replace("{semester}", str(semester))
    )
    client = DeepSeekVisionClient.from_settings()
    return client._request(
        {"prompt": prompt, "image_url": image_url},
        ManualSuggestionFields,
        {"operation": "manual_question_suggestion"},
    )


def recognize_manual_suggestion(
    image_path: str,
    bbox: list[float],
    mode: Literal["ocr", "llm"],
    *,
    subject: str | None,
    grade: int,
    semester: int,
) -> dict:
    if mode == "ocr":
        text = _ocr_text(image_path, bbox)
        fields = ManualSuggestionFields(
            instruction="",
            prompt_text=text,
            question_type="",
            correct_answer="",
            student_answer="",
        )
        return {"mode": mode, "fields": fields.model_dump(), "ocr_text": text}

    fields = _llm_fields(
        image_path,
        bbox,
        subject=subject,
        grade=grade,
        semester=semester,
    )
    return {"mode": mode, "fields": fields.model_dump(), "ocr_text": ""}
