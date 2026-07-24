"""Contradiction-only local OCR verification for proposed question crops."""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from importlib.metadata import version
from typing import Callable, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.services.question_image import load_cropped_rgb_image
from app.services.vision_recognition import VisionItem


class OCRLine(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    confidence: float = Field(ge=0, le=1)


class OCRVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["support", "contradict", "inconclusive", "unavailable"]
    matched_index: Optional[int] = None
    text_summary: str = ""
    confidence: Optional[float] = None


def _normalized_text(value: str, remove_tones: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    if remove_tones:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFKD", normalized)
            if not unicodedata.combining(character)
        )
    return "".join(character for character in normalized if character.isalnum())


def _similarity(first: str, second: str) -> float:
    variants = (
        (_normalized_text(first), _normalized_text(second)),
        (
            _normalized_text(first, remove_tones=True),
            _normalized_text(second, remove_tones=True),
        ),
    )
    return max(
        (
            SequenceMatcher(None, left, right).ratio()
            for left, right in variants
            if left and right
        ),
        default=0.0,
    )


def _item_candidate_texts(item: VisionItem) -> List[str]:
    return [
        value
        for value in (
            item.prompt_text,
            item.raw_text,
            item.normalized_text,
            item.answer,
        )
        if value
    ]


def classify_ocr_lines(
    lines: Sequence[OCRLine],
    target_index: int,
    items: Sequence[VisionItem],
    line_confidence_threshold: float,
    min_effective_characters: int,
    support_similarity_threshold: float,
    contradiction_similarity_threshold: float,
) -> OCRVerification:
    usable_lines = [
        line
        for line in lines
        if line.confidence >= line_confidence_threshold
        and len(_normalized_text(line.text, remove_tones=True))
        >= min_effective_characters
    ]
    text_summary = " | ".join(line.text for line in usable_lines)[:200]
    if not usable_lines or target_index < 0 or target_index >= len(items):
        return OCRVerification(status="inconclusive", text_summary=text_summary)

    scores = []
    for item in items:
        scores.append(
            max(
                (
                    _similarity(line.text, candidate)
                    for line in usable_lines
                    for candidate in _item_candidate_texts(item)
                ),
                default=0.0,
            )
        )

    confidence = max(line.confidence for line in usable_lines)
    target_score = scores[target_index]
    if target_score >= support_similarity_threshold:
        return OCRVerification(
            status="support",
            matched_index=target_index,
            text_summary=text_summary,
            confidence=confidence,
        )

    other_indexes = [index for index in range(len(items)) if index != target_index]
    if other_indexes:
        best_other_index = max(other_indexes, key=scores.__getitem__)
        if (
            scores[best_other_index] >= contradiction_similarity_threshold
            and scores[best_other_index] > target_score
        ):
            return OCRVerification(
                status="contradict",
                matched_index=best_other_index,
                text_summary=text_summary,
                confidence=confidence,
            )

    return OCRVerification(
        status="inconclusive",
        text_summary=text_summary,
        confidence=confidence,
    )


def _unavailable() -> OCRVerification:
    return OCRVerification(status="unavailable")


class RapidOCRVerifier:
    _engine_cache = {}

    def __init__(
        self,
        enabled: bool,
        library_version: str,
        engine_name: str,
        model_version: str,
        model_type: str,
        model_path: str,
        max_pixels: int,
        line_confidence_threshold: float,
        min_effective_characters: int,
        support_similarity_threshold: float,
        contradiction_similarity_threshold: float,
        engine_factory: Optional[Callable[[], object]] = None,
    ):
        self.enabled = enabled
        self.library_version = library_version
        self.engine_name = engine_name
        self.model_version = model_version
        self.model_type = model_type
        self.model_path = model_path
        self.max_pixels = max_pixels
        self.line_confidence_threshold = line_confidence_threshold
        self.min_effective_characters = min_effective_characters
        self.support_similarity_threshold = support_similarity_threshold
        self.contradiction_similarity_threshold = contradiction_similarity_threshold
        self.engine_factory = engine_factory

    def _create_default_engine(self):
        if version("rapidocr") != self.library_version:
            raise RuntimeError("Configured RapidOCR version is not installed")
        from rapidocr import (
            EngineType,
            LangDet,
            LangRec,
            ModelType,
            OCRVersion,
            RapidOCR,
        )

        return RapidOCR(
            params={
                "Global.model_root_dir": self.model_path,
                "Global.log_level": "warning",
                "Det.engine_type": EngineType(self.engine_name),
                "Det.lang_type": LangDet.CH,
                "Det.model_type": ModelType(self.model_type),
                "Det.ocr_version": OCRVersion(self.model_version),
                "Rec.engine_type": EngineType(self.engine_name),
                "Rec.lang_type": LangRec.CH,
                "Rec.model_type": ModelType(self.model_type),
                "Rec.ocr_version": OCRVersion(self.model_version),
            }
        )

    def _engine(self):
        if self.engine_factory is not None:
            return self.engine_factory()
        cache_key = (
            self.engine_name,
            self.library_version,
            self.model_version,
            self.model_type,
            self.model_path,
        )
        if cache_key not in self._engine_cache:
            self._engine_cache[cache_key] = self._create_default_engine()
        return self._engine_cache[cache_key]

    @staticmethod
    def _lines_from_result(result) -> List[OCRLine]:
        texts = getattr(result, "txts", None) or ()
        scores = getattr(result, "scores", None) or ()
        return [
            OCRLine(text=str(text), confidence=float(score))
            for text, score in zip(texts, scores)
            if text
        ]

    def verify(
        self,
        image_path: str,
        bbox: List[float],
        target_index: int,
        items: Sequence[VisionItem],
    ) -> OCRVerification:
        if not self.enabled:
            return _unavailable()
        try:
            import numpy

            crop = load_cropped_rgb_image(
                image_path,
                bbox,
                max_pixels=self.max_pixels,
            )
            result = self._engine()(numpy.asarray(crop), use_cls=False)
            lines = self._lines_from_result(result)
        except Exception:
            return _unavailable()

        return classify_ocr_lines(
            lines=lines,
            target_index=target_index,
            items=items,
            line_confidence_threshold=self.line_confidence_threshold,
            min_effective_characters=self.min_effective_characters,
            support_similarity_threshold=self.support_similarity_threshold,
            contradiction_similarity_threshold=self.contradiction_similarity_threshold,
        )
