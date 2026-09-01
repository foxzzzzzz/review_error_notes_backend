"""Benchmark the main branch single-pass cross-anchor vision solution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
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

from app.services.vision_recognition import MiniMaxVisionClient


SOLUTION_ID = "main_single_pass"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("vision_benchmark_config.json")
DIAGNOSTIC_PATH = Path(__file__).with_name("diagnose_vision_pipeline.py")


def _load_diagnostic_module():
    module_name = "vision_benchmark_diagnostic"
    spec = importlib.util.spec_from_file_location(module_name, DIAGNOSTIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load diagnostic script: {DIAGNOSTIC_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


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
        assignments.append(
            {
                "prediction_id": prediction["prediction_id"],
                "truth_id": matched_truth_id,
                "best_iou": round(best_iou, 6),
            }
        )
        if matched_truth_id is not None:
            by_truth[matched_truth_id].append(prediction["prediction_id"])
    ordered_truth_ids = [region["truth_id"] for region in truth]
    matched_truth_ids = [truth_id for truth_id in ordered_truth_ids if truth_id in by_truth]
    missed_truth_ids = [truth_id for truth_id in ordered_truth_ids if truth_id not in by_truth]
    false_prediction_ids = [
        assignment["prediction_id"]
        for assignment in assignments
        if assignment["truth_id"] is None
    ]
    duplicate_truth_assignments = [
        {"truth_id": truth_id, "prediction_ids": by_truth[truth_id]}
        for truth_id in ordered_truth_ids
        if len(by_truth.get(truth_id, [])) > 1
    ]
    return {
        "truth_count": len(truth),
        "prediction_count": len(predictions),
        "matched_truth_count": len(matched_truth_ids),
        "truth_recall": round(len(matched_truth_ids) / len(truth), 6),
        "matched_truth_ids": matched_truth_ids,
        "missed_truth_ids": missed_truth_ids,
        "false_prediction_ids": false_prediction_ids,
        "duplicate_truth_assignments": duplicate_truth_assignments,
        "assignments": assignments,
    }


def normalize_main_page_result(
    *,
    stable_events: dict,
    experiment_summary: dict,
    cross_cv_timing_ms: float,
) -> dict:
    run_count = experiment_summary.get("llm2_localization_run_count")
    if run_count != 1 or stable_events.get("run_count") != 1:
        raise ValueError("main benchmark requires exactly one single LLM2 pass")
    predictions = []
    for event in stable_events.get("events", []):
        question_bboxes = event.get("question_bboxes")
        if not isinstance(question_bboxes, list) or len(question_bboxes) != 1:
            raise ValueError("single-pass event must contain exactly one question bbox")
        predictions.append(
            {
                "prediction_id": f"P{len(predictions) + 1}",
                "event_id": event["event_id"],
                "cross_ids": event["cross_ids"],
                "bbox": _validate_bbox(
                    question_bboxes[0], context=f"event-{event['event_id']}"
                ),
                "confidence": event["confidence"],
            }
        )
    stage_timings = experiment_summary.get("timings_ms") or {}
    experiment_total = float(stage_timings.get("total", 0.0))
    post_audit = float(stage_timings.get("post_llm2_audit", 0.0))
    recognition_total = max(0.0, experiment_total - post_audit)
    return {
        "solution_id": SOLUTION_ID,
        "predictions": predictions,
        "llm_request_count": int(experiment_summary["llm_request_count"]),
        "timing_ms": {
            "core_total": round(cross_cv_timing_ms + recognition_total, 2),
            "cross_candidate_cv": round(cross_cv_timing_ms, 2),
            "cross_anchor_experiment": round(experiment_total, 2),
            "llm1_candidate_verification": stage_timings.get("llm1_candidate_verification"),
            "independent_cross_scan": stage_timings.get("independent_cross_scan"),
            "fallback_verification": stage_timings.get("fallback_montage_and_verification"),
            "llm2_localization": stage_timings.get("llm2_localization"),
            "post_audit": post_audit,
            "ocr": 0,
        },
        "experiment_summary": experiment_summary,
    }


def run_main_page(
    *,
    image_path: Path,
    page_dir: Path,
    subject: str,
    truth_regions: list[dict],
    cross_config: dict,
    diagnostic,
) -> dict:
    if int(cross_config.get("cross_anchor_llm2_localization_runs", 0)) != 1:
        raise ValueError("main benchmark config must use exactly one LLM2 pass")
    page_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(image_path, page_dir / ("source" + image_path.suffix.lower()))
    cv_started = time.perf_counter()
    diagnostic.write_cross_cv_artifacts(
        image_path, page_dir, cross_config, truth_regions
    )
    cross_cv_timing_ms = round((time.perf_counter() - cv_started) * 1000, 2)
    cv_dir = page_dir / "cv-cross-experiment"
    candidates = json.loads((cv_dir / "candidates.json").read_text(encoding="utf-8"))

    recorder = diagnostic.ExchangeRecorder(page_dir)
    client = MiniMaxVisionClient.from_settings()
    client.diagnostic_event_sink = recorder
    recording_client = diagnostic.RecordingVisionClient(client, recorder)
    experiment_summary = diagnostic.run_cross_anchor_experiment(
        image_path=image_path,
        case_dir=page_dir,
        client=recording_client,
        cv_candidates=candidates,
        candidate_overlay_path=cv_dir / "candidates-overlay.jpg",
        truth_regions=truth_regions,
        config=cross_config,
        subject_hint=subject,
    )
    calls_dir = page_dir / "llm-calls"
    if calls_dir.exists():
        calls_dir.rename(page_dir / "raw-llm-calls")
    experiment_dir = page_dir / "cross-anchor-experiment"
    stable_events = json.loads(
        (experiment_dir / "stable-question-events.json").read_text(encoding="utf-8")
    )
    return normalize_main_page_result(
        stable_events=stable_events,
        experiment_summary=experiment_summary,
        cross_cv_timing_ms=cross_cv_timing_ms,
    )


def _draw_overlay(image_path: Path, output_path: Path, predictions: list[dict], truth: list[dict]) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(3, image.width // 500)
    for region in truth:
        x1, y1, x2, y2 = region["source_bbox_normalized"]
        box = (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
        draw.rectangle(box, outline=(255, 210, 0), width=line_width)
        draw.text((box[0], max(0, box[1] - 14)), region["truth_id"], fill=(180, 120, 0), font=font)
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["bbox"]
        box = (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
        draw.rectangle(box, outline=(0, 150, 255), width=line_width)
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
    comparison = [
        "# 视觉方案基准报告",
        "",
        "| 图片 | 真值 | 预测 | 命中 | 召回 | 误报 | 重复归属 | LLM请求 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    timing = [
        "# 视觉方案耗时报告",
        "",
        "> 核心耗时从整页输入到错题区域输出；OCR未执行且不计时。",
        "",
        "| 图片 | 核心总耗时(ms) | 红叉候选CV(ms) | 新方案总耗时(ms) | LLM1核验(ms) | 独立扫描(ms) | fallback复核(ms) | LLM2定位(ms) | 后置审计(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in page_results:
        audit = result["truth_audit"]
        stages = result["timing_ms"]
        comparison.append(
            f"| {result['label']} | {audit['truth_count']} | {audit['prediction_count']} | "
            f"{audit['matched_truth_count']} | {audit['truth_recall']} | "
            f"{len(audit['false_prediction_ids'])} | {len(audit['duplicate_truth_assignments'])} | "
            f"{result['llm_request_count']} |"
        )
        timing.append(
            f"| {result['label']} | {stages['core_total']} | {stages['cross_candidate_cv']} | "
            f"{stages['cross_anchor_experiment']} | {stages['llm1_candidate_verification']} | "
            f"{stages['independent_cross_scan']} | {stages['fallback_verification']} | "
            f"{stages['llm2_localization']} | {stages['post_audit']} |"
        )
    (output_dir / "comparison-report.md").write_text("\n".join(comparison) + "\n", encoding="utf-8")
    (output_dir / "timing-report.md").write_text("\n".join(timing) + "\n", encoding="utf-8")


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
    benchmark_config = json.loads(args.config.read_text(encoding="utf-8"))
    schema_version = benchmark_config.get("schema_version")
    min_iou = benchmark_config.get("question_truth_min_iou")
    if not isinstance(schema_version, int) or not isinstance(min_iou, (int, float)) or not 0 < min_iou <= 1:
        raise ValueError("invalid benchmark config")
    cross_config_path = Path(benchmark_config["cross_cv_config_path"])
    if not cross_config_path.is_absolute():
        cross_config_path = args.config.parent / cross_config_path
    cross_config = json.loads(cross_config_path.read_text(encoding="utf-8"))
    if cross_config.get("cross_anchor_llm2_localization_runs") != 1:
        raise ValueError("cross CV config must use exactly one LLM2 pass")
    if float(cross_config.get("question_truth_min_iou")) != float(min_iou):
        raise ValueError("benchmark and cross CV truth IoU thresholds differ")

    labels = [label for label, _path in args.images]
    truth_by_label = load_truth_regions(args.truth_regions, labels)
    args.output.mkdir(parents=True, exist_ok=False)
    diagnostic = _load_diagnostic_module()
    page_results = []
    image_hashes = {}
    for label, image_path in args.images:
        page_dir = args.output / "pages" / label
        result = run_main_page(
            image_path=image_path,
            page_dir=page_dir,
            subject=args.subject,
            truth_regions=truth_by_label[label],
            cross_config=cross_config,
            diagnostic=diagnostic,
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
        _draw_overlay(
            image_path,
            page_dir / "annotated-predictions.jpg",
            result["predictions"],
            truth_by_label[label],
        )

    total_truth = sum(result["truth_audit"]["truth_count"] for result in page_results)
    total_matched = sum(result["truth_audit"]["matched_truth_count"] for result in page_results)
    summary = {
        "schema_version": schema_version,
        "solution_id": SOLUTION_ID,
        "truth_sha256": _sha256(args.truth_regions),
        "image_sha256_by_label": image_hashes,
        "all_truth_recalled": total_matched == total_truth,
        "truth_count": total_truth,
        "matched_truth_count": total_matched,
        "truth_recall": round(total_matched / total_truth, 6),
        "false_prediction_count": sum(len(result["truth_audit"]["false_prediction_ids"]) for result in page_results),
        "llm_request_count": sum(result["llm_request_count"] for result in page_results),
        "core_timing_ms": round(sum(result["timing_ms"]["core_total"] for result in page_results), 2),
        "pages": [
            {
                "label": result["label"],
                "truth_audit": result["truth_audit"],
                "llm_request_count": result["llm_request_count"],
                "timing_ms": result["timing_ms"],
            }
            for result in page_results
        ],
    }
    manifest = {
        "schema_version": schema_version,
        "solution_id": SOLUTION_ID,
        "git_commit": _git_commit(),
        "benchmark_config_sha256": _sha256(args.config),
        "cross_cv_config_sha256": _sha256(cross_config_path),
        "truth_sha256": summary["truth_sha256"],
        "image_sha256_by_label": image_hashes,
    }
    _write_json(args.output / "manifest.json", manifest)
    _write_json(args.output / "summary.json", summary)
    _write_reports(args.output, page_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
