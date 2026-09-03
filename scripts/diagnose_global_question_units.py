"""Diagnose deterministic page-global question units without LLM requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Literal

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict, Field


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.local_ocr_verification import RapidOCRVerifier
from app.services.vision_recognition import (
    MiniMaxVisionClient,
    prepare_image_data_url,
)
from global_question_units import (
    audit_fixed_units,
    build_global_question_units,
    compare_unit_candidates_to_truth,
    load_config,
    map_anchors_to_units,
    validate_bbox,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("global_question_unit_config.json")


SEMANTIC_JUDGE_PROMPT = """你是小学作业错题语义裁判。图片按C编号展示红叉锚点及其允许选择的U题目单元；红框是待核验锚点，绿色框是本地CV+OCR生成的稳定题目单元。

你的任务不是画框，而是理解批改语义：核验红叉真假、判断红叉属于哪道题、结合学生实际答题判断是否为错题，并检查候选单元是否完整且没有侵入兄弟题。

要求：
1. 每个输入cross_id必须且只能返回一次，不得遗漏、增加、合并或重排。
2. selected_unit_id只能从该cross_id的allowed_units中选择一个unit_id；图内R编号与visual_rank对应，不能跨锚点选择。
3. 红叉是主要证据；附近红圈、老师批注、改正痕迹和答案不匹配可以辅助判断，但印刷红格不能单独证明错题。
4. anchor_validity分别为valid、invalid、uncertain；question_status分别为incorrect、correct、uncertain。不得采用全部默认确认策略，必须逐个检查红框内是否有两条相交红色斜线及实际答题情况。
5. boundary_fit分别为complete、too_narrow、sibling_intrusion、uncertain。绿色框应包含一整道独立作答单元，不能把左右或上下兄弟题一起算入。
6. 只有明确对应错题时填写selected_unit_id；不是红叉或不是错题时返回null，无法判断时使用uncertain并返回null。
7. supplemental_wrong_unit_ids只能引用输入candidate_unit_ids中的编号，用于补充本页明显错题但未被逐锚点结果选中的单元；没有则返回空数组。
8. evidence只能从red_cross、red_circle、teacher_correction、answer_mismatch、student_correction、insufficient_detail、other中选择。
9. 不得返回或生成任何bbox、坐标、新编号、题目文字、解释或Markdown，只返回严格JSON。

返回格式：{"decisions":[{"cross_id":0,"anchor_validity":"valid","selected_unit_id":"U-S01-R01-C01","question_status":"incorrect","boundary_fit":"complete","evidence":["red_cross","answer_mismatch"],"confidence":0.93}],"supplemental_wrong_unit_ids":[]}。

科目：__SUBJECT__
输入候选：__CANDIDATES__
"""


class SemanticAnchorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cross_id: int
    anchor_validity: Literal["valid", "invalid", "uncertain"]
    selected_unit_id: str | None = None
    question_status: Literal["incorrect", "correct", "uncertain"]
    boundary_fit: Literal[
        "complete", "too_narrow", "sibling_intrusion", "uncertain"
    ]
    evidence: list[
        Literal[
            "red_cross",
            "red_circle",
            "teacher_correction",
            "answer_mismatch",
            "student_correction",
            "insufficient_detail",
            "other",
        ]
    ]
    confidence: float = Field(ge=0, le=1)


class SemanticJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decisions: list[SemanticAnchorDecision]
    supplemental_wrong_unit_ids: list[str]


def audit_semantic_judgment(
    decisions: list[dict], supplemental_unit_ids: list[str], mapping: dict
) -> dict:
    expected_ids = {int(value) for value in mapping["anchor_candidates"]}
    counts = {cross_id: 0 for cross_id in expected_ids}
    allowed = {
        int(cross_id): {
            candidate["question_unit_id"] for candidate in candidates
        }
        for cross_id, candidates in mapping["anchor_candidates"].items()
    }
    violations = []
    for decision in decisions:
        cross_id = int(decision["cross_id"])
        if cross_id not in expected_ids:
            violations.append({"cross_id": cross_id, "reason": "unknown_anchor"})
            continue
        counts[cross_id] += 1
    for cross_id in sorted(expected_ids):
        if counts[cross_id] == 0:
            violations.append({"cross_id": cross_id, "reason": "missing_anchor"})
        elif counts[cross_id] > 1:
            violations.append({"cross_id": cross_id, "reason": "duplicate_anchor"})
    accepted = []
    for decision in decisions:
        cross_id = int(decision["cross_id"])
        if cross_id not in expected_ids or counts[cross_id] != 1:
            continue
        selected = decision.get("selected_unit_id")
        if selected is not None and selected not in allowed[cross_id]:
            violations.append(
                {"cross_id": cross_id, "reason": "unit_not_allowed_for_anchor"}
            )
            continue
        selected_expected = (
            decision.get("anchor_validity") == "valid"
            and decision.get("question_status") == "incorrect"
        )
        if selected_expected != (selected is not None):
            violations.append(
                {"cross_id": cross_id, "reason": "inconsistent_selection_state"}
            )
            continue
        accepted.append(decision)
    candidate_pool = set().union(*allowed.values()) if allowed else set()
    accepted_supplemental = []
    for unit_id in supplemental_unit_ids:
        if unit_id not in candidate_pool:
            violations.append(
                {"unit_id": unit_id, "reason": "unknown_supplemental_unit"}
            )
        elif unit_id not in accepted_supplemental:
            accepted_supplemental.append(unit_id)
    return {
        "accepted": sorted(accepted, key=lambda item: int(item["cross_id"])),
        "accepted_supplemental_unit_ids": sorted(accepted_supplemental),
        "violations": violations,
    }


def build_semantic_unit_sets(audit: dict, mapping: dict) -> dict:
    accepted_by_id = {
        int(item["cross_id"]): item for item in audit["accepted"]
    }
    strict = set(audit["accepted_supplemental_unit_ids"])
    needs_review = []
    for raw_cross_id, candidates in sorted(
        mapping["anchor_candidates"].items(), key=lambda item: int(item[0])
    ):
        cross_id = int(raw_cross_id)
        decision = accepted_by_id.get(cross_id)
        selected = decision.get("selected_unit_id") if decision else None
        if selected is not None:
            strict.add(selected)
            continue
        fallback = candidates[0]["question_unit_id"] if candidates else None
        if decision is None:
            reason = "unassigned_anchor" if not candidates else "semantic_missing_or_invalid"
        elif (
            decision["anchor_validity"] == "uncertain"
            or decision["question_status"] == "uncertain"
        ):
            reason = "semantic_uncertain"
        elif decision["anchor_validity"] == "invalid":
            reason = "semantic_rejected"
        else:
            reason = "semantic_correct"
        needs_review.append(
            {"cross_id": cross_id, "fallback_unit_id": fallback, "reason": reason}
        )
    recall_safe = strict | {
        item["fallback_unit_id"]
        for item in needs_review
        if item["fallback_unit_id"] is not None
    }
    return {
        "strict_unit_ids": sorted(strict),
        "recall_safe_unit_ids": sorted(recall_safe),
        "needs_review": needs_review,
    }


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _parse_images(values: list[str]) -> list[tuple[str, Path]]:
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


def _resolve_anchor_path(root: Path, label: str) -> Path:
    candidates = (
        root / label / "cross-anchor-experiment" / "confirmed-crosses.json",
        root / "pages" / label / "cross-anchor-experiment" / "confirmed-crosses.json",
    )
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one confirmed-crosses replay for {label}")
    return matches[0]


def _load_truth(path: Path, labels: list[str]) -> dict[str, list[dict]]:
    pages = json.loads(path.read_text(encoding="utf-8")).get("pages")
    if not isinstance(pages, dict):
        raise ValueError("truth JSON must contain pages")
    truth = {}
    for label in labels:
        regions = pages.get(label, {}).get("regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"truth page has no regions: {label}")
        truth[label] = regions
    return truth


def _ocr_verifier() -> RapidOCRVerifier:
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


def _pixel_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(bbox[0] * width),
        round(bbox[1] * height),
        round(bbox[2] * width),
        round(bbox[3] * height),
    )


def _write_overlays(
    *, image_path: Path, page_dir: Path, units: list[dict], anchors: list[dict],
    mapping: dict
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot load image: {image_path}")
    height, width = image.shape[:2]
    unit_overlay = image.copy()
    for unit in units:
        bbox = _pixel_bbox(validate_bbox(unit["unit_bbox"]), width, height)
        cv2.rectangle(unit_overlay, bbox[:2], bbox[2:], (0, 180, 0), 3)
        cv2.putText(
            unit_overlay, unit["question_unit_id"], (bbox[0] + 3, bbox[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(page_dir / "global-question-units-overlay.jpg"), unit_overlay)
    anchor_overlay = unit_overlay.copy()
    unit_by_id = {unit["question_unit_id"]: unit for unit in units}
    for anchor in anchors:
        cross_id = int(anchor["cross_id"])
        bbox = _pixel_bbox(validate_bbox(anchor["bbox"]), width, height)
        cv2.rectangle(anchor_overlay, bbox[:2], bbox[2:], (0, 0, 255), 3)
        for candidate in mapping["anchor_candidates"][str(cross_id)]:
            unit = unit_by_id[candidate["question_unit_id"]]
            candidate_bbox = _pixel_bbox(unit["unit_bbox"], width, height)
            cv2.putText(
                anchor_overlay,
                f"C{cross_id}:R{candidate['rank']}",
                (candidate_bbox[0] + 3, candidate_bbox[3] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA,
            )
    cv2.imwrite(
        str(page_dir / "anchor-unit-candidates-overlay.jpg"), anchor_overlay
    )


def write_semantic_judge_montage(
    *, image_path: Path, output_path: Path, units: list[dict], anchors: list[dict],
    mapping: dict, config: dict
) -> None:
    with Image.open(image_path) as source:
        page = ImageOps.exif_transpose(source).convert("RGB")
    columns = int(config["semantic_montage_columns"])
    tile_width = int(config["semantic_montage_tile_width"])
    tile_height = int(config["semantic_montage_tile_height"])
    label_height = int(config["semantic_montage_label_height"])
    font_size = int(config["semantic_montage_font_size"])
    padding = float(config["semantic_montage_crop_padding_ratio"])
    rows = max(1, (len(anchors) + columns - 1) // columns)
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    unit_by_id = {unit["question_unit_id"]: unit for unit in units}
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    for index, anchor in enumerate(sorted(anchors, key=lambda item: int(item["cross_id"]))):
        cross_id = int(anchor["cross_id"])
        candidates = mapping["anchor_candidates"][str(cross_id)]
        boxes = [validate_bbox(anchor["bbox"])] + [
            validate_bbox(unit_by_id[item["question_unit_id"]]["context_bbox"])
            for item in candidates
        ]
        crop_bbox = [
            max(0.0, min(box[0] for box in boxes) - padding),
            max(0.0, min(box[1] for box in boxes) - padding),
            min(1.0, max(box[2] for box in boxes) + padding),
            min(1.0, max(box[3] for box in boxes) + padding),
        ]
        pixel_crop = _pixel_bbox(crop_bbox, page.width, page.height)
        crop = page.crop(pixel_crop)
        crop_width = max(crop_bbox[2] - crop_bbox[0], 1e-9)
        crop_height = max(crop_bbox[3] - crop_bbox[1], 1e-9)

        available_height = tile_height - label_height
        crop.thumbnail((tile_width, available_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(crop)

        def local_box(bbox):
            return (
                round((bbox[0] - crop_bbox[0]) / crop_width * crop.width),
                round((bbox[1] - crop_bbox[1]) / crop_height * crop.height),
                round((bbox[2] - crop_bbox[0]) / crop_width * crop.width),
                round((bbox[3] - crop_bbox[1]) / crop_height * crop.height),
            )

        draw.rectangle(local_box(validate_bbox(anchor["bbox"])), outline="red", width=5)
        for candidate in candidates:
            unit = unit_by_id[candidate["question_unit_id"]]
            bbox = local_box(validate_bbox(unit["unit_bbox"]))
            draw.rectangle(bbox, outline="green", width=4)
            draw.text(
                (bbox[0] + 3, bbox[1] + 3),
                f"R{candidate['rank']}",
                fill="green",
                font=font,
            )
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(
            crop,
            ((tile_width - crop.width) // 2, label_height + (available_height - crop.height) // 2),
        )
        tile_draw = ImageDraw.Draw(tile)
        rank_text = "/".join(f"R{item['rank']}" for item in candidates) or "none"
        tile_draw.text(
            (8, 6), f"C{cross_id} candidates: {rank_text}", fill="black", font=font
        )
        tile_draw.rectangle((0, 0, tile_width - 1, tile_height - 1), outline="black")
        montage.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    montage.save(
        output_path,
        quality=int(config["semantic_montage_jpeg_quality"]),
    )


def _geometry_fingerprint(unit_result: dict, mapping: dict) -> str:
    canonical = json.dumps(
        {
            "units": unit_result["units"],
            "anchor_candidates": mapping["anchor_candidates"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def run_page(
    *, label: str, image_path: Path, anchor_path: Path, truth_regions: list[dict],
    config: dict, config_sha256: str, output_root: Path,
    run_semantic_judge: bool, subject: str
) -> dict:
    started = time.perf_counter()
    page_dir = output_root / label
    page_dir.mkdir(parents=True, exist_ok=False)
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    for anchor in anchors:
        validate_bbox(anchor.get("bbox"))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot load image: {image_path}")
    ocr_started = time.perf_counter()
    page_ocr = _ocr_verifier().recognize_page(
        str(image_path), int(config["ocr_full_page_max_edge"])
    )
    ocr_ms = (time.perf_counter() - ocr_started) * 1000
    ocr_lines = []
    if page_ocr.status == "available":
        ocr_lines = [
            {**line.model_dump(mode="json"), "ocr_line_id": index}
            for index, line in enumerate(page_ocr.lines)
        ]
    layout_started = time.perf_counter()
    unit_result = build_global_question_units(image, ocr_lines, config)
    layout_ms = (time.perf_counter() - layout_started) * 1000
    mapping_started = time.perf_counter()
    mapping = map_anchors_to_units(unit_result["units"], anchors, config)
    anchor_mapping_ms = (time.perf_counter() - mapping_started) * 1000
    audit_started = time.perf_counter()
    audit = compare_unit_candidates_to_truth(
        unit_result["units"], mapping, truth_regions, config
    )
    audit_ms = (time.perf_counter() - audit_started) * 1000
    ablations = {}
    for mode in ("grid_only", "ocr_only", "combined"):
        result = build_global_question_units(image, ocr_lines, config, evidence_mode=mode)
        mode_mapping = map_anchors_to_units(result["units"], anchors, config)
        mode_audit = compare_unit_candidates_to_truth(
            result["units"], mode_mapping, truth_regions, config
        )
        ablations[mode] = {
            "unit_count": len(result["units"]),
            "truth_recall": mode_audit["truth_recall"],
            "missed_truth_ids": mode_audit["missed_truth_ids"],
            "sibling_intrusion_unit_ids": mode_audit["sibling_intrusion_unit_ids"],
        }
    _write_json(page_dir / "global-question-units.json", unit_result)
    _write_json(page_dir / "anchor-unit-candidates.json", mapping)
    _write_json(page_dir / "question-unit-oracle-audit.json", audit)
    _write_json(page_dir / "ablation-audit.json", ablations)
    _write_overlays(
        image_path=image_path, page_dir=page_dir, units=unit_result["units"],
        anchors=anchors, mapping=mapping,
    )
    candidate_ids = {
        candidate["question_unit_id"]
        for candidates in mapping["anchor_candidates"].values()
        for candidate in candidates
    }
    semantic_montage_path = page_dir / "semantic-judge-montage.jpg"
    semantic_response = {"decisions": [], "supplemental_wrong_unit_ids": []}
    semantic_audit = {
        "accepted": [],
        "accepted_supplemental_unit_ids": [],
        "violations": [],
    }
    semantic_error = None
    llm_events = []
    llm_ms = 0.0
    llm_request_count = 0
    if run_semantic_judge:
        write_semantic_judge_montage(
            image_path=image_path,
            output_path=semantic_montage_path,
            units=unit_result["units"],
            anchors=anchors,
            mapping=mapping,
            config=config,
        )
        prompt_payload = {
            "anchors": [
                {
                    "cross_id": int(anchor["cross_id"]),
                    "source": anchor.get("source"),
                    "cv_confidence": anchor.get("confidence"),
                    "allowed_units": [
                        {
                            "visual_rank": f"R{item['rank']}",
                            "unit_id": item["question_unit_id"],
                        }
                        for item in mapping["anchor_candidates"].get(
                            str(int(anchor["cross_id"])), []
                        )
                    ],
                }
                for anchor in sorted(anchors, key=lambda item: int(item["cross_id"]))
            ],
            "candidate_unit_ids": sorted(candidate_ids),
        }
        prompt = SEMANTIC_JUDGE_PROMPT.replace("__SUBJECT__", subject).replace(
            "__CANDIDATES__",
            json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        )
        client = MiniMaxVisionClient.from_settings()
        client.diagnostic_event_sink = llm_events.append
        diagnostic_context = {
            "operation": "global_question_unit_semantic_judge",
            "label": label,
            "anchor_count": len(anchors),
            "candidate_unit_count": len(candidate_ids),
        }
        llm_started = time.perf_counter()
        llm_request_count = 1
        try:
            result = client._request(
                {
                    "prompt": prompt,
                    "image_url": prepare_image_data_url(
                        str(semantic_montage_path),
                        client.max_edge,
                        client.jpeg_quality,
                        diagnostic_context,
                    ),
                },
                SemanticJudgeResult,
                diagnostic_context,
            )
            semantic_response = result.model_dump(mode="json")
            semantic_audit = audit_semantic_judgment(
                semantic_response["decisions"],
                semantic_response["supplemental_wrong_unit_ids"],
                mapping,
            )
        except Exception as exc:
            semantic_error = {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "message": str(exc),
                "diagnostic": getattr(exc, "diagnostic", None),
            }
        llm_ms = (time.perf_counter() - llm_started) * 1000
    semantic_sets = build_semantic_unit_sets(semantic_audit, mapping)
    unit_by_id = {unit["question_unit_id"]: unit for unit in unit_result["units"]}
    strict_units = [
        unit_by_id[unit_id]
        for unit_id in semantic_sets["strict_unit_ids"]
        if unit_id in unit_by_id
    ]
    recall_safe_units = [
        unit_by_id[unit_id]
        for unit_id in semantic_sets["recall_safe_unit_ids"]
        if unit_id in unit_by_id
    ]
    strict_audit = audit_fixed_units(strict_units, truth_regions, config)
    recall_safe_audit = audit_fixed_units(recall_safe_units, truth_regions, config)
    _write_json(page_dir / "semantic-judge-response.json", semantic_response)
    _write_json(page_dir / "semantic-judge-audit.json", semantic_audit)
    _write_json(page_dir / "semantic-unit-sets.json", semantic_sets)
    _write_json(page_dir / "semantic-strict-comparison.json", strict_audit)
    _write_json(
        page_dir / "semantic-recall-safe-comparison.json", recall_safe_audit
    )
    _write_json(page_dir / "semantic-judge-error.json", semantic_error)
    _write_json(page_dir / "llm-events.json", llm_events)
    summary = {
        "label": label,
        "status": "completed",
        "truth_count": len(truth_regions),
        "unit_count": len(unit_result["units"]),
        "anchor_count": len(anchors),
        "unassigned_anchor_count": len(mapping["unassigned_anchors"]),
        "candidate_unit_count": len(candidate_ids),
        "candidate_oracle_truth_recall": audit["truth_recall"],
        "candidate_oracle_missed_truth_ids": audit["missed_truth_ids"],
        "false_unit_count": len(audit["false_unit_ids"]),
        "sibling_intrusion_unit_count": len(audit["sibling_intrusion_unit_ids"]),
        "config_sha256": config_sha256,
        "geometry_fingerprint": _geometry_fingerprint(unit_result, mapping),
        "ocr_status": page_ocr.status,
        "semantic_status": (
            "not_run"
            if not run_semantic_judge
            else ("llm_error" if semantic_error else "completed")
        ),
        "semantic_strict_truth_recall": strict_audit["truth_recall"],
        "semantic_strict_false_unit_count": len(strict_audit["false_unit_ids"]),
        "semantic_recall_safe_truth_recall": recall_safe_audit["truth_recall"],
        "semantic_recall_safe_false_unit_count": len(
            recall_safe_audit["false_unit_ids"]
        ),
        "semantic_none_or_uncertain_count": sum(
            item.get("selected_unit_id") is None
            for item in semantic_audit["accepted"]
        ),
        "semantic_violation_count": len(semantic_audit["violations"]),
        "semantic_needs_review_count": len(semantic_sets["needs_review"]),
        "ocr_ms": round(ocr_ms, 2),
        "layout_ms": round(layout_ms, 2),
        "anchor_mapping_ms": round(anchor_mapping_ms, 2),
        "audit_ms": round(audit_ms, 2),
        "llm_ms": round(llm_ms, 2),
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
        "llm_request_count": llm_request_count,
    }
    _write_json(page_dir / "summary.json", summary)
    return summary


def _write_report(path: Path, summaries: list[dict]) -> None:
    lines = [
        "# 全局题目单元阶段A诊断",
        "",
        "> 全流程仅使用本地CV与OCR；真值只参与后置审计，LLM请求恒为0。",
        "",
        "| 图片 | 真值 | 单元 | 锚点 | 未分配锚点 | 候选召回 | 候选误报 | 模型严格召回 | 模型严格误报 | 安全召回 | 安全误报 | needs_review | 语义异常 | 兄弟题侵入 | OCR(ms) | 版面(ms) | LLM(ms) | 总耗时(ms) | LLM请求 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {label} | {truth_count} | {unit_count} | {anchor_count} | "
            "{unassigned_anchor_count} | {candidate_oracle_truth_recall} | "
            "{false_unit_count} | {semantic_strict_truth_recall} | "
            "{semantic_strict_false_unit_count} | "
            "{semantic_recall_safe_truth_recall} | "
            "{semantic_recall_safe_false_unit_count} | "
            "{semantic_needs_review_count} | {semantic_violation_count} | "
            "{sibling_intrusion_unit_count} | {ocr_ms} | {layout_ms} | "
            "{llm_ms} | {total_ms} | {llm_request_count} |".format(**item)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(arguments=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Images in label=/absolute/path form")
    parser.add_argument("--anchors-root", required=True, type=Path)
    parser.add_argument("--truth-regions", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-semantic-judge", action="store_true")
    parser.add_argument("--subject", default="chinese")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(arguments)
    images = _parse_images(args.images)
    labels = [label for label, _ in images]
    if len(labels) != len(set(labels)):
        parser.error("image labels must be unique")
    anchor_root = args.anchors_root.expanduser().resolve()
    if not anchor_root.is_dir():
        parser.error("--anchors-root must be an existing directory")
    truth_path = args.truth_regions.expanduser().resolve()
    truth = _load_truth(truth_path, labels)
    config_path = args.config.expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = load_config(config_path)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    summaries = [
        run_page(
            label=label,
            image_path=image_path,
            anchor_path=_resolve_anchor_path(anchor_root, label),
            truth_regions=truth[label],
            config=config,
            config_sha256=config_sha256,
            output_root=output,
            run_semantic_judge=args.run_semantic_judge,
            subject=args.subject,
        )
        for label, image_path in images
    ]
    aggregate = {
        "experiment": "global_question_unit_stage_a",
        "offline_only": True,
        "labels": labels,
        "llm_request_count": sum(item["llm_request_count"] for item in summaries),
        "truth_count": sum(item["truth_count"] for item in summaries),
        "page_summaries": summaries,
        "config_sha256": config_sha256,
    }
    _write_json(output / "summary.json", aggregate)
    _write_report(output / "comparison-report.md", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
