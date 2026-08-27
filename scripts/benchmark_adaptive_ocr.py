"""Benchmark adaptive red scanning and full-page RapidOCR without logging OCR text."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
from importlib.metadata import version

from app.config import settings
from app.services.error_mark_validation import scan_red_mark_regions
from app.services.local_ocr_verification import RapidOCRVerifier
from app.services.vision_recognition import VisionItem


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile_value)))
    return ordered[index]


def representative_question_bbox(mark_bbox: list[float]) -> list[float]:
    center_x = (mark_bbox[0] + mark_bbox[2]) / 2
    center_y = (mark_bbox[1] + mark_bbox[3]) / 2
    width = max(0.4, mark_bbox[2] - mark_bbox[0])
    height = max(0.18, mark_bbox[3] - mark_bbox[1])
    left = min(max(0.0, center_x - width / 2), 1.0 - width)
    top = min(max(0.0, center_y - height / 2), 1.0 - height)
    return [left, top, left + width, top + height]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    verifier = RapidOCRVerifier(
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
    red_times = []
    ocr_times = []
    crop_times = []
    images = []
    benchmark_item = VisionItem(
        raw_text="基准",
        instruction="基准测试",
        prompt_text="基准",
        normalized_text="基准",
        answer="基准",
        subject="chinese",
        question_type="other",
        tags=[],
        difficulty=1,
        confidence=1.0,
        uncertain_segments=[],
    )
    for image_path in args.images:
        scan = scan_red_mark_regions(
            image_path,
            max_edge=settings.LOCAL_RED_SCAN_MAX_EDGE,
            min_component_pixels=settings.LOCAL_RED_COMPONENT_MIN_PIXELS,
            max_component_area_ratio=settings.LOCAL_RED_COMPONENT_MAX_AREA_RATIO,
            max_thinness_ratio=settings.LOCAL_RED_COMPONENT_MAX_THINNESS_RATIO,
        )
        red_times.append(scan.duration_ms)
        image_result = {
            "file": image_path.rsplit("/", 1)[-1],
            "red_status": scan.status,
            "red_region_count": len(scan.regions),
            "red_scan_ms": round(scan.duration_ms, 2),
            "page_ocr_ms": [],
            "marked_crop_ocr_ms": [],
        }
        for _ in range(args.repeats):
            page = verifier.recognize_page(
                image_path,
                max_edge=settings.LOCAL_OCR_FULL_PAGE_MAX_EDGE,
            )
            if page.status != "available":
                raise RuntimeError(f"page OCR unavailable: {page.error_code or page.status}")
            ocr_times.append(page.duration_ms)
            image_result["page_ocr_ms"].append(round(page.duration_ms, 2))
        if scan.regions:
            bbox = representative_question_bbox(scan.regions[0].bbox)
            for _ in range(args.repeats):
                verification = verifier.verify_crop(
                    image_path,
                    bbox,
                    target_index=0,
                    items=[benchmark_item],
                )
                if verification.status == "unavailable":
                    raise RuntimeError(
                        f"crop OCR unavailable: {verification.error_code}"
                    )
                crop_times.append(verification.duration_ms)
                image_result["marked_crop_ocr_ms"].append(
                    round(verification.duration_ms, 2)
                )
        images.append(image_result)

    output = {
        "cpu": platform.processor() or platform.machine(),
        "rapidocr_version": version("rapidocr"),
        "onnxruntime_version": version("onnxruntime"),
        "ocr_enabled": settings.LOCAL_OCR_ENABLED,
        "worker_concurrency": settings.CELERY_WORKER_CONCURRENCY,
        "full_page_max_edge": settings.LOCAL_OCR_FULL_PAGE_MAX_EDGE,
        "image_count": len(args.images),
        "ocr_call_count": len(ocr_times),
        "red_scan_p50_ms": round(statistics.median(red_times), 2),
        "red_scan_p95_ms": round(percentile(red_times, 0.95), 2),
        "page_ocr_p50_ms": round(statistics.median(ocr_times), 2),
        "page_ocr_p95_ms": round(percentile(ocr_times, 0.95), 2),
        "marked_crop_ocr_p50_ms": (
            round(statistics.median(crop_times), 2) if crop_times else None
        ),
        "marked_crop_ocr_p95_ms": (
            round(percentile(crop_times, 0.95), 2) if crop_times else None
        ),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "images": images,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
