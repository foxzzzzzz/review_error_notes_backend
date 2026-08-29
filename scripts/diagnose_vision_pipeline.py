"""Capture CV and three-stage vision evidence without persisting questions.

The output contains worksheet images and model responses. Store it only in a
restricted diagnostic directory and delete it after analysis.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps
import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.error_mark_validation import (
    group_red_evidence_regions,
    scan_red_mark_regions,
)
from app.services.vision_recognition import (
    MiniMaxVisionClient,
    recognize_marked_three_stage,
)


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
    with Image.open(image_path) as source:
        mask_source = ImageOps.exif_transpose(source).convert("RGB")
        if max(mask_source.size) > max_edge:
            mask_source.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    pixels = np.asarray(mask_source, dtype=np.int16)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    red_mask = (
        (red >= 120)
        & (red - green >= 45)
        & (red - blue >= 45)
        & (red >= green * 1.35)
        & (red >= blue * 1.35)
    )
    Image.fromarray((red_mask * 255).astype(np.uint8), mode="L").save(
        output_dir / "cv" / "red-mask.png"
    )
    _draw_boxes(
        image_path,
        output_dir / "cv" / "components-and-groups.jpg",
        [("C", components, "red"), ("G", grouped, "blue")],
    )
    return {**payload, "scan_result": scan}


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


def build_summary(
    *,
    label: str,
    expected_count: int | None,
    cv: dict,
    pipeline: dict | None,
) -> dict:
    pipeline_ran = pipeline is not None
    pipeline = pipeline or {}
    checkpoints = {
        "expected_error_count": expected_count,
        "cv_raw_component_count": cv.get("raw_component_count"),
        "cv_evidence_group_count": cv.get("evidence_group_count"),
        "llm_mark_primitive_count": pipeline.get("mark_primitive_count"),
        "normalized_mark_event_count": pipeline.get("mark_event_count"),
        "localized_mark_count": pipeline.get("localized_mark_count"),
        "content_item_count": pipeline.get("content_item_count"),
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
) -> dict:
    case_dir = output_dir / label
    case_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(image_path, case_dir / ("source" + image_path.suffix.lower()))
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
    pipeline_diagnostic = None
    error = None
    if not cv_only:
        recorder = ExchangeRecorder(case_dir)
        client = MiniMaxVisionClient.from_settings()
        client.diagnostic_event_sink = recorder
        recording_client = RecordingVisionClient(client, recorder)
        arguments = _pipeline_arguments()
        _write_json(case_dir / "effective-config.json", arguments)
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
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "message": str(exc),
                "diagnostic": getattr(exc, "diagnostic", None),
            }
            pipeline_diagnostic = getattr(exc, "diagnostic", None)
            _write_json(case_dir / "pipeline-error.json", error)
    summary = build_summary(
        label=label,
        expected_count=expected_count,
        cv=cv,
        pipeline=pipeline_diagnostic,
    )
    summary["error"] = error
    _write_json(case_dir / "summary.json", summary)
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


def _write_report(output_dir: Path, summaries: list[dict]) -> None:
    lines = [
        "# 视觉识别流程诊断报告",
        "",
        "> CV 组件/证据组数量不等于错题数量；本表只用于定位首次数量偏差。",
        "",
        "| 图片 | 人工错题 | CV组件 | CV证据组 | 标准化红标事件 | 定位 | 内容 | 首次数量偏差 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        item = summary["checkpoints"]
        divergence = (
            "未运行"
            if summary["pipeline_status"] == "not_run"
            else summary["first_count_divergence"] or "无数量偏差"
        )
        lines.append(
            "| {label} | {expected} | {components} | {groups} | {marks} | {located} | {content} | {divergence} |".format(
                label=summary["label"],
                expected=item["expected_error_count"],
                components=item["cv_raw_component_count"],
                groups=item["cv_evidence_group_count"],
                marks=item["normalized_mark_event_count"],
                located=item["localized_mark_count"],
                content=item["content_item_count"],
                divergence=divergence,
            )
        )
    (output_dir / "comparison-report.md").write_text(
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
    args = parser.parse_args()

    try:
        images = _parse_labeled_paths(args.images)
        expected = _parse_expected(args.expected)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        output_dir / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "warning": "Contains worksheet images, prompts, and model responses; delete after analysis.",
            "cv_only": args.cv_only,
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
            cv_only=args.cv_only,
        )
        for label, path in images
    ]
    _write_json(output_dir / "summary.json", summaries)
    _write_report(output_dir, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 1 if any(summary["error"] for summary in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
