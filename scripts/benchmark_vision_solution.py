"""Benchmark the branch-native wrong-question region recognition flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.services.error_mark_validation import (
    ErrorMarkImageInvalid,
    filter_valid_error_marks,
)
from app.services.vision_recognition import (
    MiniMaxVisionClient,
    VisionRecognitionError,
    bbox_area,
    localization_matches_evidence,
    localization_passes_geometry,
    marker_focused_display_bbox,
    validated_localizations,
)


SOLUTION_ID = "old_solution"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("vision_benchmark_config.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_bbox(value, *, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{context} must be a four-number bbox")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{context} must contain numbers")
    bbox = [float(item) for item in value]
    if not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
        raise ValueError(f"{context} must be an ordered normalized bbox")
    return bbox


def load_truth_regions(path: Path, labels: list[str]) -> dict[str, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = payload.get("pages")
    if not isinstance(pages, dict):
        raise ValueError("truth-regions.json must contain a pages object")
    loaded = {}
    for label in labels:
        page = pages.get(label)
        if not isinstance(page, dict) or not isinstance(page.get("regions"), list):
            raise ValueError(f"missing truth page: {label}")
        seen = set()
        regions = []
        for region in page["regions"]:
            truth_id = region.get("truth_id")
            if not isinstance(truth_id, str) or not truth_id or truth_id in seen:
                raise ValueError(f"invalid or duplicate truth_id for {label}: {truth_id}")
            seen.add(truth_id)
            regions.append(
                {
                    **region,
                    "source_bbox_normalized": _validate_bbox(
                        region.get("source_bbox_normalized"),
                        context=f"{label}/{truth_id}",
                    ),
                }
            )
        if not regions:
            raise ValueError(f"truth page has no regions: {label}")
        loaded[label] = regions
    return loaded


def _bbox_iou(left: list[float], right: list[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def compare_predictions_to_truth(
    predictions: list[dict], truth: list[dict], *, min_iou: float
) -> dict:
    assignments = []
    by_truth = defaultdict(list)
    for prediction in predictions:
        ranked = sorted(
            (
                (
                    _bbox_iou(prediction["bbox"], region["source_bbox_normalized"]),
                    region["truth_id"],
                )
                for region in truth
            ),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        best_iou, truth_id = ranked[0] if ranked else (0.0, None)
        matched_truth_id = truth_id if best_iou >= min_iou else None
        assignment = {
            "prediction_id": prediction["prediction_id"],
            "truth_id": matched_truth_id,
            "best_iou": round(best_iou, 6),
        }
        assignments.append(assignment)
        if matched_truth_id is not None:
            by_truth[matched_truth_id].append(prediction["prediction_id"])

    ordered_truth_ids = [region["truth_id"] for region in truth]
    matched_truth_ids = [truth_id for truth_id in ordered_truth_ids if truth_id in by_truth]
    missed_truth_ids = [truth_id for truth_id in ordered_truth_ids if truth_id not in by_truth]
    false_prediction_ids = [
        item["prediction_id"] for item in assignments if item["truth_id"] is None
    ]
    duplicate_truth_assignments = [
        {"truth_id": truth_id, "prediction_ids": prediction_ids}
        for truth_id, prediction_ids in (
            (truth_id, by_truth.get(truth_id, [])) for truth_id in ordered_truth_ids
        )
        if len(prediction_ids) > 1
    ]
    truth_count = len(truth)
    return {
        "truth_count": truth_count,
        "prediction_count": len(predictions),
        "matched_truth_count": len(matched_truth_ids),
        "truth_recall": round(len(matched_truth_ids) / truth_count, 6),
        "matched_truth_ids": matched_truth_ids,
        "missed_truth_ids": missed_truth_ids,
        "false_prediction_ids": false_prediction_ids,
        "duplicate_truth_assignments": duplicate_truth_assignments,
        "assignments": assignments,
    }


class RecordingMiniMaxVisionClient(MiniMaxVisionClient):
    """Capture HTTP attempts without persisting credentials or base64 image data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_request_records = []

    def _post(self, payload):
        image_url = payload.get("image_url", "")
        record = {
            "request_index": len(self.benchmark_request_records) + 1,
            "prompt": payload.get("prompt"),
            "image_url_sha256": hashlib.sha256(image_url.encode("utf-8")).hexdigest(),
            "image_url_character_count": len(image_url),
        }
        started = time.perf_counter()
        self.benchmark_request_records.append(record)
        try:
            response = super()._post(payload)
            record["status_code"] = response.status_code
            try:
                record["raw_response"] = response.json()
            except ValueError:
                record["raw_response_text"] = response.text
            return response
        except Exception as exc:
            record["request_error"] = type(exc).__name__
            raise
        finally:
            record["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)

    def _request(self, payload, result_model, diagnostic):
        first_record = len(self.benchmark_request_records)
        try:
            result = super()._request(payload, result_model, diagnostic)
        except Exception:
            for record in self.benchmark_request_records[first_record:]:
                record["operation"] = diagnostic.get("operation")
            raise
        records = self.benchmark_request_records[first_record:]
        for record in records:
            record["operation"] = diagnostic.get("operation")
        if records:
            records[-1]["validated_response"] = result.model_dump(mode="json")
        return result


def run_old_solution_page(
    *,
    image_path: Path,
    subject: str,
    client,
    mark_confidence_threshold: float,
    red_pixel_min_ratio: float,
    red_pixel_expansion_ratio: float,
    localization_threshold: float = 0.0,
    localization_max_area_ratio: float = 1.0,
    crop_context_padding_ratio: float = 0.0,
) -> dict:
    total_started = time.perf_counter()
    first_record = len(getattr(client, "benchmark_request_records", []))

    phase_started = time.perf_counter()
    recognition = client.recognize(str(image_path), subject_hint=subject)
    recognition_ms = round((time.perf_counter() - phase_started) * 1000, 2)

    phase_started = time.perf_counter()
    try:
        valid_marks, rejected_mark_ids, mark_audit = filter_valid_error_marks(
            str(image_path),
            recognition.error_marks,
            confidence_threshold=mark_confidence_threshold,
            red_pixel_min_ratio=red_pixel_min_ratio,
            expansion_ratio=red_pixel_expansion_ratio,
        )
    except ErrorMarkImageInvalid:
        valid_marks = []
        rejected_mark_ids = [mark.mark_id for mark in recognition.error_marks]
        mark_audit = []
    mark_filter_ms = round((time.perf_counter() - phase_started) * 1000, 2)

    phase_started = time.perf_counter()
    localization_result = None
    localizations = {}
    if not recognition.error_marks or valid_marks:
        try:
            localization_result = client.localize(
                str(image_path), recognition.items, valid_marks
            )
            localizations = validated_localizations(
                localization_result,
                item_count=len(recognition.items),
                marks={mark.mark_id: mark for mark in valid_marks},
            )
        except VisionRecognitionError:
            localization_result = None
            localizations = {}
    localization_ms = round((time.perf_counter() - phase_started) * 1000, 2)

    marks_by_id = {mark.mark_id: mark for mark in valid_marks}
    predictions = []
    for index, item in enumerate(recognition.items):
        localization = localizations.get(index)
        if (
            localization is None
            or localization.confidence < localization_threshold
            or not localization_passes_geometry(
                localization,
                marks=marks_by_id,
                max_area_ratio=localization_max_area_ratio,
            )
            or not localization_matches_evidence(localization, item)
        ):
            continue
        display_bbox = marker_focused_display_bbox(
            localization_bbox=localization.bbox,
            mark_ids=localization.mark_ids,
            marks=marks_by_id,
            padding_ratio=crop_context_padding_ratio,
        )
        predictions.append(
            {
                "prediction_id": f"P{len(predictions) + 1}",
                "item_index": index,
                "bbox": display_bbox,
                "localization_bbox": localization.bbox,
                "mark_ids": localization.mark_ids,
                "confidence": localization.confidence,
                "observed_prompt_text": localization.observed_prompt_text,
                "observed_raw_text": localization.observed_raw_text,
            }
        )

    request_records = list(
        getattr(client, "benchmark_request_records", [])[first_record:]
    )
    logical_request_count = 2
    total_ms = round((time.perf_counter() - total_started) * 1000, 2)
    return {
        "solution_id": SOLUTION_ID,
        "predictions": predictions,
        "llm_request_count": len(request_records) or logical_request_count,
        "request_records": request_records,
        "timing_ms": {
            "core_total": total_ms,
            "recognition": recognition_ms,
            "mark_filter_cv": mark_filter_ms,
            "localization": localization_ms,
            "post_audit": round(
                max(0.0, total_ms - recognition_ms - mark_filter_ms - localization_ms),
                2,
            ),
            "ocr": 0,
        },
        "recognition": recognition.model_dump(mode="json"),
        "valid_error_mark_ids": [mark.mark_id for mark in valid_marks],
        "rejected_error_mark_ids": rejected_mark_ids,
        "error_mark_validation": mark_audit,
        "localization": (
            localization_result.model_dump(mode="json")
            if localization_result is not None
            else None
        ),
    }


def run_old_solution_page_safely(
    *,
    image_path: Path,
    subject: str,
    client,
    mark_confidence_threshold: float,
    red_pixel_min_ratio: float,
    red_pixel_expansion_ratio: float,
    localization_threshold: float = 0.0,
    localization_max_area_ratio: float = 1.0,
    crop_context_padding_ratio: float = 0.0,
) -> dict:
    page_started = time.perf_counter()
    first_record = len(getattr(client, "benchmark_request_records", []))
    try:
        result = run_old_solution_page(
            image_path=image_path,
            subject=subject,
            client=client,
            mark_confidence_threshold=mark_confidence_threshold,
            red_pixel_min_ratio=red_pixel_min_ratio,
            red_pixel_expansion_ratio=red_pixel_expansion_ratio,
            localization_threshold=localization_threshold,
            localization_max_area_ratio=localization_max_area_ratio,
            crop_context_padding_ratio=crop_context_padding_ratio,
        )
    except Exception as exc:
        diagnostic = getattr(exc, "diagnostic", None) or {}
        request_records = list(
            getattr(client, "benchmark_request_records", [])[first_record:]
        )
        return {
            "solution_id": SOLUTION_ID,
            "page_status": "failed",
            "failed_stage": diagnostic.get("operation") or "unknown",
            "error": {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "message": str(exc),
                "diagnostic": diagnostic,
            },
            "predictions": [],
            "llm_request_count": len(request_records),
            "request_records": request_records,
            "timing_ms": {
                "core_total": round(
                    (time.perf_counter() - page_started) * 1000, 2
                ),
                "recognition": None,
                "mark_filter_cv": None,
                "localization": None,
                "post_audit": None,
                "ocr": 0,
            },
            "recognition": None,
            "valid_error_mark_ids": [],
            "rejected_error_mark_ids": [],
            "error_mark_validation": [],
            "localization": None,
        }
    return {**result, "page_status": "completed", "failed_stage": None, "error": None}


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _draw_overlay(image_path: Path, output_path: Path, predictions: list[dict], truth: list[dict]) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for region in truth:
        x1, y1, x2, y2 = region["source_bbox_normalized"]
        box = (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
        draw.rectangle(box, outline=(255, 210, 0), width=max(3, image.width // 500))
        draw.text((box[0], max(0, box[1] - 14)), region["truth_id"], fill=(180, 120, 0), font=font)
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["bbox"]
        box = (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
        draw.rectangle(box, outline=(0, 150, 255), width=max(3, image.width // 500))
        draw.text((box[0], box[1]), prediction["prediction_id"], fill=(0, 80, 200), font=font)
    image.save(output_path, format="JPEG", quality=94)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_reports(output_dir: Path, page_results: list[dict]) -> None:
    comparison_lines = [
        "# 视觉方案基准报告",
        "",
        "| 图片 | 状态 | 失败阶段 | 真值 | 预测 | 命中 | 召回 | 误报 | 重复归属 | LLM请求 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    timing_lines = [
        "# 视觉方案耗时报告",
        "",
        "> 核心耗时从整页输入到错题区域输出；OCR未执行且不计时。",
        "",
        "| 图片 | 核心总耗时(ms) | 识别LLM(ms) | 红色证据CV(ms) | 定位LLM(ms) | 后置审计(ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in page_results:
        audit = result["truth_audit"]
        timing = result["timing_ms"]
        comparison_lines.append(
            f"| {result['label']} | {result['page_status']} | {result['failed_stage'] or '-'} | "
            f"{audit['truth_count']} | {audit['prediction_count']} | "
            f"{audit['matched_truth_count']} | {audit['truth_recall']} | "
            f"{len(audit['false_prediction_ids'])} | {len(audit['duplicate_truth_assignments'])} | "
            f"{result['llm_request_count']} |"
        )
        timing_lines.append(
            f"| {result['label']} | {timing['core_total']} | {timing['recognition']} | "
            f"{timing['mark_filter_cv']} | {timing['localization']} | {timing['post_audit']} |"
        )
    (output_dir / "comparison-report.md").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")
    (output_dir / "timing-report.md").write_text("\n".join(timing_lines) + "\n", encoding="utf-8")


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Images in label=/absolute/path form")
    parser.add_argument("--truth-regions", required=True, type=Path)
    parser.add_argument("--subject", default="chinese")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(arguments)
    parsed_images = []
    seen = set()
    for value in args.images:
        if "=" not in value:
            parser.error(f"invalid image argument: {value}")
        label, raw_path = value.split("=", 1)
        if not label or label in seen:
            parser.error(f"invalid or duplicate image label: {label}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            parser.error(f"image not found: {path}")
        seen.add(label)
        parsed_images.append((label, path))
    args.images = parsed_images
    return args


def main(arguments: Iterable[str] | None = None) -> int:
    args = parse_args(arguments)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    schema_version = config.get("schema_version")
    min_iou = config.get("question_truth_min_iou")
    if not isinstance(schema_version, int) or not isinstance(min_iou, (int, float)) or not 0 < min_iou <= 1:
        raise ValueError("invalid benchmark config")
    labels = [label for label, _path in args.images]
    truth_by_label = load_truth_regions(args.truth_regions, labels)
    args.output.mkdir(parents=True, exist_ok=False)

    client = RecordingMiniMaxVisionClient.from_settings()
    page_results = []
    image_hashes = {}
    for label, image_path in args.images:
        page_dir = args.output / "pages" / label
        raw_dir = page_dir / "raw-llm-calls"
        raw_dir.mkdir(parents=True)
        result = run_old_solution_page_safely(
            image_path=image_path,
            subject=args.subject,
            client=client,
            mark_confidence_threshold=settings.MINIMAX_MARK_CONFIDENCE_THRESHOLD,
            red_pixel_min_ratio=settings.MARK_RED_PIXEL_MIN_RATIO,
            red_pixel_expansion_ratio=settings.MARK_RED_PIXEL_EXPANSION_RATIO,
            localization_threshold=settings.MINIMAX_LOCALIZATION_CONFIDENCE_THRESHOLD,
            localization_max_area_ratio=settings.MINIMAX_LOCALIZATION_MAX_AREA_RATIO,
            crop_context_padding_ratio=settings.QUESTION_CROP_CONTEXT_PADDING_RATIO,
        )
        truth_audit = compare_predictions_to_truth(
            result["predictions"], truth_by_label[label], min_iou=float(min_iou)
        )
        page_result = {**result, "label": label, "truth_audit": truth_audit}
        page_results.append(page_result)
        image_hashes[label] = _sha256(image_path)
        _write_json(page_dir / "predictions.json", result["predictions"])
        _write_json(page_dir / "truth-audit.json", truth_audit)
        _write_json(page_dir / "timing.json", result["timing_ms"])
        _write_json(page_dir / "branch-native-result.json", {
            key: value for key, value in result.items() if key not in {"request_records", "predictions", "timing_ms"}
        })
        for record in result["request_records"]:
            _write_json(raw_dir / f"request-{record['request_index']:03d}.json", record)
        _draw_overlay(
            image_path,
            page_dir / "annotated-predictions.jpg",
            result["predictions"],
            truth_by_label[label],
        )

    total_truth = sum(item["truth_audit"]["truth_count"] for item in page_results)
    total_matched = sum(item["truth_audit"]["matched_truth_count"] for item in page_results)
    summary = {
        "schema_version": schema_version,
        "solution_id": SOLUTION_ID,
        "truth_sha256": _sha256(args.truth_regions),
        "image_sha256_by_label": image_hashes,
        "all_truth_recalled": total_matched == total_truth,
        "failed_page_count": sum(
            item["page_status"] == "failed" for item in page_results
        ),
        "failed_pages": [
            {
                "label": item["label"],
                "failed_stage": item["failed_stage"],
                "error": item["error"],
            }
            for item in page_results
            if item["page_status"] == "failed"
        ],
        "truth_count": total_truth,
        "matched_truth_count": total_matched,
        "truth_recall": round(total_matched / total_truth, 6),
        "false_prediction_count": sum(len(item["truth_audit"]["false_prediction_ids"]) for item in page_results),
        "llm_request_count": sum(item["llm_request_count"] for item in page_results),
        "core_timing_ms": round(sum(item["timing_ms"]["core_total"] for item in page_results), 2),
        "pages": [
            {
                "label": item["label"],
                "page_status": item["page_status"],
                "failed_stage": item["failed_stage"],
                "error": item["error"],
                "truth_audit": item["truth_audit"],
                "llm_request_count": item["llm_request_count"],
                "timing_ms": item["timing_ms"],
            }
            for item in page_results
        ],
    }
    manifest = {
        "schema_version": schema_version,
        "solution_id": SOLUTION_ID,
        "git_commit": _git_commit(),
        "benchmark_config_sha256": _sha256(args.config),
        "truth_sha256": summary["truth_sha256"],
        "image_sha256_by_label": image_hashes,
    }
    _write_json(args.output / "manifest.json", manifest)
    _write_json(args.output / "summary.json", summary)
    _write_reports(args.output, page_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
