#!/usr/bin/env python3
"""Render per-stage bbox diagnostics for marked-evidence validation pages."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


COLORS = (
    "#e53935",
    "#1e88e5",
    "#43a047",
    "#fb8c00",
    "#8e24aa",
    "#00897b",
    "#6d4c41",
)


def _valid_bbox(value) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        bbox = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError):
        return None
    if not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
        return None
    return bbox


def _draw_overlay(
    image_path: Path,
    output_path: Path,
    items: Iterable[dict],
    *,
    bbox_keys=("bbox",),
    label_key="id",
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    line_width = max(2, round(max(width, height) / 700))
    for index, item in enumerate(items):
        color = COLORS[index % len(COLORS)]
        label = str(item.get(label_key, index))
        for key in bbox_keys:
            bbox = _valid_bbox(item.get(key))
            if bbox is None:
                continue
            pixel_bbox = (
                round(bbox[0] * width),
                round(bbox[1] * height),
                round(bbox[2] * width),
                round(bbox[3] * height),
            )
            draw.rectangle(pixel_bbox, outline=color, width=line_width)
            text = f"{label}:{key}" if len(bbox_keys) > 1 else label
            text_bbox = draw.textbbox((pixel_bbox[0], pixel_bbox[1]), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((pixel_bbox[0], pixel_bbox[1]), text, fill="white", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _stage_audit_for(image: dict) -> dict:
    for question in image.get("questions") or []:
        raw = question.get("ocr_raw_json") or {}
        audit = raw.get("stage_audit")
        if isinstance(audit, dict):
            return audit
    raise ValueError(
        f"image {image.get('image_id')} has no stage_audit; "
        "start the shadow worker with CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED=true"
    )


def _persisted_candidates(image: dict) -> list[dict]:
    candidates = []
    for index, question in enumerate(image.get("questions") or []):
        raw = question.get("ocr_raw_json") or {}
        identity = ((raw.get("evidence_bundle") or {}).get("identity") or {})
        geometry = identity.get("question_geometry") or {}
        candidates.append(
            {
                "question_id": question.get("id"),
                "mark_id": identity.get("mark_id"),
                "bbox": geometry.get("question_bbox") or geometry.get("bbox"),
                "index": index,
            }
        )
    return candidates


def _content_observations(image: dict) -> list[dict]:
    observations = []
    seen = set()

    def visit(value):
        if isinstance(value, dict):
            bbox = _valid_bbox(value.get("bbox"))
            if bbox is not None and "text" in value and "source" in value:
                key = (str(value.get("source")), str(value.get("text")), tuple(bbox))
                if key not in seen:
                    seen.add(key)
                    observations.append(
                        {
                            "id": len(observations),
                            "source": value.get("source"),
                            "role": value.get("role"),
                            "text": value.get("text"),
                            "bbox": bbox,
                        }
                    )
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for question in image.get("questions") or []:
        visit((question.get("ocr_raw_json") or {}).get("evidence_bundle") or {})
    return observations


def _candidate_ledger(image: dict) -> list[dict]:
    """Return review-only primary-page evidence without making a quality claim."""
    ledger = []
    for question in image.get("questions") or []:
        raw = question.get("ocr_raw_json") or {}
        crop_region = question.get("crop_region") or {}
        bundle = raw.get("evidence_bundle") or {}
        cv_audit = raw.get("local_cv_audit") or {}
        cv_status = cv_audit.get("status") or ("available" if cv_audit else "unavailable")
        ocr_page = raw.get("ocr_page") or {}
        ledger.append(
            {
                "question_id": question.get("id"),
                "content": {
                    "printed_question": question.get("ocr_text"),
                    "student_answer": question.get("ocr_answer"),
                },
                "model_bbox": raw.get("model_bbox") or crop_region.get("model_bbox"),
                "display_bbox": raw.get("display_bbox") or crop_region.get("display_bbox"),
                "display_bbox_scale": raw.get("display_bbox_scale")
                or crop_region.get("display_bbox_scale"),
                "ocr": {
                    "status": raw.get("ocr_advisory_status", "unavailable"),
                    "page_status": ocr_page.get("status", "not_provided"),
                    "conflicts": bundle.get("role_conflicts") or [],
                },
                "cv": {
                    "status": cv_status,
                    "covered_region_indexes": cv_audit.get("covered_region_indexes") or [],
                    "uncovered_region_indexes": (
                        cv_audit.get("page_uncovered_region_indexes") or []
                    ),
                },
                "timing": raw.get("evidence_timing") or {},
            }
        )
    return ledger


def _write_summary_files(output_dir: Path, summary: list[dict]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "label",
        "image_id",
        "truth_count",
        "raw_component_count",
        "evidence_group_count",
        "mark_attempt_primitive_count",
        "mark_event_count",
        "localized_count",
        "content_item_count",
        "persisted_candidate_count",
        "event_inflation",
        "red_scan_ms",
        "full_page_ocr_ms",
        "mark_detection_ms",
        "question_localization_ms",
        "content_recognition_ms",
        "before_persistence_elapsed_ms",
        "pre_commit_elapsed_ms",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# 分阶段膨胀审计",
        "",
        "| 页面 | 人工真值 | 红色组件 | 证据组 | 两轮标记图元 | 归并事件 | 题区 | 内容 | 持久化 | 事件膨胀 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {label} | {truth_count} | {raw_component_count} | "
            "{evidence_group_count} | {mark_attempt_primitive_count} | "
            "{mark_event_count} | {localized_count} | {content_item_count} | "
            "{persisted_candidate_count} | {event_inflation} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 分阶段耗时（毫秒）",
            "",
            "| 页面 | 红色扫描 | 全页OCR | 标记检测 | 题区定位 | 内容识别 | 持久化前累计 | 提交前累计 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            "| {label} | {red_scan_ms} | {full_page_ocr_ms} | "
            "{mark_detection_ms} | {question_localization_ms} | "
            "{content_recognition_ms} | {before_persistence_elapsed_ms} | "
            "{pre_commit_elapsed_ms} |".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_stage_audit_report(
    *, review_path: Path, output_dir: Path, pages: list[dict], image_audits: dict | None = None
) -> list[dict]:
    review_images = json.loads(review_path.read_text(encoding="utf-8"))
    by_image_id = {image["image_id"]: image for image in review_images}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for page in pages:
        image = by_image_id.get(page["image_id"])
        image_audit = (image_audits or {}).get(str(page["image_id"])) or {}
        if image is None:
            if not image_audit:
                raise ValueError(f"review data has no image_id {page['image_id']}")
            image = {"image_id": page["image_id"], "questions": []}
        image_path = Path(page["image_path"])
        if not image_path.is_file():
            raise ValueError(f"source image does not exist: {image_path}")
        audit = _stage_audit_for(image) if image.get("questions") else {}
        page_dir = output_dir / str(page["label"])
        page_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(image_audit.get("page_primary_raw_response"), str):
            (page_dir / "primary-raw-response.md").write_text(
                image_audit["page_primary_raw_response"], encoding="utf-8"
            )
            (page_dir / "primary-raw-response.json").write_text(
                json.dumps(image_audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        source_suffix = Path(str(page.get("source_name") or image_path.name)).suffix.lower()
        shutil.copyfile(image_path, page_dir / f"00-source{source_suffix}")

        red_components = audit.get("red_components") or []
        evidence_groups = audit.get("evidence_groups") or []
        ocr_lines = audit.get("ocr_lines") or []
        mark_attempts = audit.get("mark_attempts") or []
        merged_primitives = audit.get("merged_primitives") or []
        mark_events = audit.get("mark_events") or []
        localizations = audit.get("localizations") or []
        content_observations = _content_observations(image)
        persisted = _persisted_candidates(image)

        _draw_overlay(
            image_path,
            page_dir / "01-red-components.png",
            [dict(item, id=f"C{item.get('component_id')}") for item in red_components],
        )
        _draw_overlay(
            image_path,
            page_dir / "02-red-groups.png",
            [dict(item, id=f"R{item.get('region_id')}") for item in evidence_groups],
        )
        _draw_overlay(
            image_path,
            page_dir / "03-ocr-lines.png",
            [dict(item, id=f"O{item.get('ocr_line_id')}") for item in ocr_lines],
        )
        for attempt in mark_attempts:
            _draw_overlay(
                image_path,
                page_dir / f"04-mark-attempt-{attempt.get('attempt')}.png",
                [
                    dict(item, id=item.get("source_ref", index))
                    for index, item in enumerate(attempt.get("marks") or [])
                ],
                bbox_keys=("bbox", "cross_bbox", "circle_bbox"),
            )
        _draw_overlay(
            image_path,
            page_dir / "05-merged-primitives.png",
            [dict(item, id=f"M{item.get('mark_id')}") for item in merged_primitives],
            bbox_keys=("bbox", "cross_bbox", "circle_bbox"),
        )
        _draw_overlay(
            image_path,
            page_dir / "06-merged-events.png",
            [dict(item, id=f"E{item.get('mark_id')}") for item in mark_events],
            bbox_keys=("bbox", "cross_bbox", "circle_bbox"),
        )
        _draw_overlay(
            image_path,
            page_dir / "07-question-localizations.png",
            [dict(item, id=f"Q{item.get('mark_id')}") for item in localizations],
            bbox_keys=("bbox", "answer_bbox", "prompt_bbox"),
        )
        _draw_overlay(
            image_path,
            page_dir / "08-content-observations.png",
            content_observations,
        )
        _draw_overlay(
            image_path,
            page_dir / "09-persisted-candidates.png",
            [dict(item, id=f"P{item.get('mark_id')}") for item in persisted],
        )

        page_audit = {
            **audit,
            "content_observations": content_observations,
            "persisted_candidates": persisted,
        }
        (page_dir / "audit.json").write_text(
            json.dumps(page_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (page_dir / "candidate-ledger.json").write_text(
            json.dumps(_candidate_ledger(image), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        first_raw = (image.get("questions") or [{}])[0].get("ocr_raw_json") or {}
        three_stage = first_raw.get("three_stage") or {}
        local_ocr_page = first_raw.get("local_ocr_page") or {}
        evidence_timing = first_raw.get("evidence_timing") or {}
        truth_count = page.get("truth_count")
        event_count = len(mark_events)
        summary.append(
            {
                "label": str(page["label"]),
                "image_id": page["image_id"],
                "truth_count": truth_count,
                "raw_component_count": len(red_components),
                "evidence_group_count": len(evidence_groups),
                "mark_attempt_primitive_count": sum(
                    len(attempt.get("marks") or []) for attempt in mark_attempts
                ),
                "mark_event_count": event_count,
                "localized_count": len(localizations),
                "content_item_count": int(three_stage.get("content_item_count", 0) or 0),
                "persisted_candidate_count": len(persisted),
                "event_inflation": (
                    event_count - int(truth_count) if truth_count is not None else None
                ),
                "red_scan_ms": audit.get("red_scan_ms"),
                "full_page_ocr_ms": local_ocr_page.get("duration_ms"),
                "mark_detection_ms": three_stage.get("mark_llm_ms"),
                "question_localization_ms": three_stage.get("localization_llm_ms"),
                "content_recognition_ms": three_stage.get("content_llm_ms"),
                "before_persistence_elapsed_ms": (
                    (evidence_timing.get("before_persistence") or {}).get("elapsed_ms")
                ),
                "pre_commit_elapsed_ms": (
                    (evidence_timing.get("pre_commit") or {}).get("elapsed_ms")
                ),
            }
        )
    _write_summary_files(output_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-images", required=True, type=Path)
    parser.add_argument("--pages-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-audits", type=Path)
    args = parser.parse_args()
    pages = json.loads(args.pages_json.read_text(encoding="utf-8"))
    image_audits = json.loads(args.image_audits.read_text(encoding="utf-8")) if args.image_audits else None
    render_stage_audit_report(
        review_path=args.review_images,
        output_dir=args.output_dir,
        pages=pages,
        image_audits=image_audits,
    )


if __name__ == "__main__":
    main()
