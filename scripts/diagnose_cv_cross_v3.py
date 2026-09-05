"""Compare the frozen cross detector with recentered deterministic CV guards."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.diagnose_vision_pipeline import (
    compare_cross_candidates_to_truth,
    detect_red_cross_candidates,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _analysis_pixels(image_path: Path, max_edge: int) -> np.ndarray:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def _ray_continuity(
    mask: np.ndarray,
    center_x: int,
    center_y: int,
    angle: float,
    direction: int,
    inner_radius: int,
    outer_radius: int,
    band: int,
) -> float:
    height, width = mask.shape
    unit_x = math.cos(angle) * direction
    unit_y = math.sin(angle) * direction
    perpendicular_x = -unit_y
    perpendicular_y = unit_x
    hits = 0
    samples = 0
    for radius in range(inner_radius, outer_radius + 1):
        samples += 1
        found = False
        for offset in range(-band, band + 1):
            x = round(center_x + unit_x * radius + perpendicular_x * offset)
            y = round(center_y + unit_y * radius + perpendicular_y * offset)
            if 0 <= x < width and 0 <= y < height and mask[y, x]:
                found = True
                break
        hits += int(found)
    return hits / samples if samples else 0.0


def opposite_arm_continuity(
    mask: np.ndarray,
    center_x: int,
    center_y: int,
    config: dict,
) -> float:
    scale = max(mask.shape)
    inner_radius = max(
        1, round(scale * float(config["continuity_inner_radius_ratio"]))
    )
    outer_radius = max(
        inner_radius + 1,
        round(scale * float(config["continuity_outer_radius_ratio"])),
    )
    band = max(1, round(scale * float(config["continuity_band_ratio"])))
    minimum_angle = int(config["continuity_angle_min_degrees"])
    maximum_angle = int(config["continuity_angle_max_degrees"])
    step = int(config["continuity_angle_step_degrees"])

    line_scores = []
    for start, stop in (
        (minimum_angle, maximum_angle),
        (180 - maximum_angle, 180 - minimum_angle),
    ):
        best = 0.0
        for degrees in range(start, stop + 1, step):
            angle = math.radians(degrees)
            best = max(
                best,
                min(
                    _ray_continuity(
                        mask,
                        center_x,
                        center_y,
                        angle,
                        1,
                        inner_radius,
                        outer_radius,
                        band,
                    ),
                    _ray_continuity(
                        mask,
                        center_x,
                        center_y,
                        angle,
                        -1,
                        inner_radius,
                        outer_radius,
                        band,
                    ),
                ),
            )
        line_scores.append(best)
    return round(min(line_scores), 6)


def _comparison_summary(comparison: dict) -> dict:
    return {
        "candidate_count": comparison["candidate_count"],
        "matched_truth_count": comparison["matched_truth_count"],
        "truth_recall": comparison["truth_recall"],
        "false_candidate_count": len(comparison["false_candidate_ids"]),
        "missed_truth_ids": comparison["missed_truth_ids"],
    }


def _draw_candidates(
    image_path: Path,
    output_path: Path,
    candidates: list[dict],
    color: tuple[int, int, int],
) -> None:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    stroke = max(2, round(max(width, height) / 700))
    for candidate in candidates:
        left, top, right, bottom = candidate["bbox"]
        box = (left * width, top * height, right * width, bottom * height)
        draw.rectangle(box, outline=color, width=stroke)
        draw.text(
            (box[0] + stroke, box[1] + stroke),
            str(candidate["candidate_id"]),
            fill=color,
        )
    image.save(output_path, quality=92)


def _truth_pages(payload: dict) -> dict:
    return payload["pages"] if "pages" in payload else payload


def _candidate_intersection_points(
    normalized_red: np.ndarray,
    red_mask: np.ndarray,
    candidate: dict,
    config: dict,
) -> list[tuple[int, int]]:
    height, width = red_mask.shape
    left, top, right, bottom = candidate["bbox"]
    x0 = max(0, min(width - 1, round(left * width)))
    y0 = max(0, min(height - 1, round(top * height)))
    x1 = max(x0 + 1, min(width, round(right * width)))
    y1 = max(y0 + 1, min(height, round(bottom * height)))
    crop_mask = red_mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(crop_mask)
    if not len(xs):
        center_x = max(0, min(width - 1, round(candidate["center"][0] * width)))
        center_y = max(0, min(height - 1, round(candidate["center"][1] * height)))
        return [(center_x, center_y)]

    absolute_xs = xs + x0
    absolute_ys = ys + y0
    points = []
    order = np.argsort(normalized_red[absolute_ys, absolute_xs])[::-1]
    minimum_spacing = max(
        1, round(max(red_mask.shape) * float(config["intersection_min_spacing_ratio"]))
    )
    limit = int(config["intersection_candidate_limit"])
    for index in order:
        point = (int(absolute_xs[index]), int(absolute_ys[index]))
        if all(
            (point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2
            >= minimum_spacing**2
            for other in points
        ):
            points.append(point)
        if len(points) >= limit:
            break
    return points[:limit]


def _best_intersection(
    normalized_red: np.ndarray,
    red_mask: np.ndarray,
    candidate: dict,
    config: dict,
) -> tuple[float, list[float]]:
    best_score = -1.0
    best_point = None
    for center_x, center_y in _candidate_intersection_points(
        normalized_red, red_mask, candidate, config
    ):
        score = opposite_arm_continuity(
            red_mask, center_x, center_y, config
        )
        if score > best_score:
            best_score = score
            best_point = (center_x, center_y)
    height, width = red_mask.shape
    return round(max(best_score, 0.0), 6), [
        round(best_point[0] / width, 6),
        round(best_point[1] / height, 6),
    ]


def filter_cross_candidates_v3(
    pixels: np.ndarray,
    candidates: list[dict],
    config: dict,
) -> dict:
    float_pixels = pixels.astype(np.float32)
    background = np.percentile(
        float_pixels.reshape(-1, 3),
        float(config["background_percentile"]),
        axis=0,
    )
    normalized = float_pixels / np.maximum(background, 1.0)
    normalized_red = normalized[:, :, 0] - np.maximum(
        normalized[:, :, 1], normalized[:, :, 2]
    )
    red_mask = normalized_red >= float(
        config["normalized_red_intersection_threshold"]
    )
    height, width = red_mask.shape
    weight = float(config["continuity_score_weight"])
    minimum_quality = float(config["minimum_candidate_quality_score"])
    kept = []
    rejected = []
    audits = []

    for candidate in candidates:
        left, top, right, bottom = candidate["bbox"]
        x0 = max(0, min(width - 1, round(left * width)))
        y0 = max(0, min(height - 1, round(top * height)))
        x1 = max(x0 + 1, min(width, round(right * width)))
        y1 = max(y0 + 1, min(height, round(bottom * height)))
        red_q99 = round(
            float(np.percentile(normalized_red[y0:y1, x0:x1], 99)), 6
        )
        continuity, best_center = _best_intersection(
            normalized_red, red_mask, candidate, config
        )
        quality = round(red_q99 + weight * continuity, 6)
        reasons = (
            []
            if quality >= minimum_quality
            else ["insufficient_combined_cross_evidence"]
        )
        audit = {
            "candidate_id": candidate["candidate_id"],
            "reasons": reasons,
            "normalized_red_q99": red_q99,
            "best_opposite_arm_continuity": continuity,
            "best_intersection_center": best_center,
            "candidate_quality_score": quality,
        }
        audits.append(audit)
        if reasons:
            rejected.append(audit)
        else:
            kept.append({**candidate, "v3_quality": audit})
    return {
        "candidates": kept,
        "rejected_candidates": rejected,
        "candidate_audits": audits,
        "background_rgb": [round(float(value), 3) for value in background],
    }


def compare_v3_with_baseline(
    baseline_candidates: list[dict],
    filtered: dict,
    truth_regions: list[dict],
    *,
    truth_match_margin_ratio: float,
) -> dict:
    baseline = compare_cross_candidates_to_truth(
        baseline_candidates, truth_regions, margin_ratio=truth_match_margin_ratio
    )
    v3 = compare_cross_candidates_to_truth(
        filtered["candidates"], truth_regions, margin_ratio=truth_match_margin_ratio
    )
    truth_by_candidate = {
        item["candidate_id"]: item["truth_id"] for item in baseline["assignments"]
    }
    rejected_ids = {item["candidate_id"] for item in filtered["rejected_candidates"]}
    return {
        "baseline": _comparison_summary(baseline),
        "v3": _comparison_summary(v3),
        "removed_true_candidate_ids": sorted(
            candidate_id
            for candidate_id in rejected_ids
            if truth_by_candidate[candidate_id] is not None
        ),
        "removed_false_candidate_ids": sorted(
            candidate_id
            for candidate_id in rejected_ids
            if truth_by_candidate[candidate_id] is None
        ),
    }


def run_case(
    label: str,
    image_path: Path,
    output_dir: Path,
    baseline_config: dict,
    v3_config: dict,
    truth_regions: list[dict],
) -> dict:
    case_dir = output_dir / label
    case_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    baseline_started = time.perf_counter()
    detected = detect_red_cross_candidates(image_path, baseline_config)
    baseline_ms = (time.perf_counter() - baseline_started) * 1000
    pixels = _analysis_pixels(image_path, int(baseline_config["analysis_max_edge"]))
    filter_started = time.perf_counter()
    filtered = filter_cross_candidates_v3(pixels, detected["candidates"], v3_config)
    filter_ms = (time.perf_counter() - filter_started) * 1000
    comparison = compare_v3_with_baseline(
        detected["candidates"],
        filtered,
        truth_regions,
        truth_match_margin_ratio=float(baseline_config["truth_match_margin_ratio"]),
    )
    _write_json(case_dir / "baseline-candidates.json", detected["candidates"])
    _write_json(case_dir / "v3-candidates.json", filtered["candidates"])
    _write_json(case_dir / "v3-rejected-candidates.json", filtered["rejected_candidates"])
    _write_json(case_dir / "v3-candidate-audit.json", filtered["candidate_audits"])
    _write_json(case_dir / "comparison.json", comparison)
    _draw_candidates(
        image_path,
        case_dir / "baseline-candidates-overlay.jpg",
        detected["candidates"],
        (0, 90, 255),
    )
    _draw_candidates(
        image_path,
        case_dir / "v3-candidates-overlay.jpg",
        filtered["candidates"],
        (0, 180, 0),
    )
    candidates_by_id = {item["candidate_id"]: item for item in detected["candidates"]}
    rejected_candidates = [
        candidates_by_id[item["candidate_id"]]
        for item in filtered["rejected_candidates"]
    ]
    _draw_candidates(
        image_path,
        case_dir / "v3-rejected-overlay.jpg",
        rejected_candidates,
        (220, 30, 30),
    )
    return {
        "label": label,
        **comparison,
        "timings_ms": {
            "baseline_detector": round(baseline_ms, 2),
            "v3_filter": round(filter_ms, 2),
            "total": round((time.perf_counter() - started) * 1000, 2),
        },
    }


def write_report(output_dir: Path, summaries: list[dict]) -> None:
    lines = [
        "# 本地红叉 CV v3 回归报告",
        "",
        "> 全程不调用LLM；v3只在冻结baseline候选上执行相交点重定位和确定性综合色度过滤。",
        "",
        "| 图片 | 真值 | baseline候选 | baseline召回 | baseline误报 | v3候选 | v3召回 | v3误报 | 删除真候选 | 删除误报 | v3耗时(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        baseline = item["baseline"]
        v3 = item["v3"]
        truth_count = baseline["matched_truth_count"] + len(baseline["missed_truth_ids"])
        lines.append(
            f"| {item['label']} | {truth_count} | {baseline['candidate_count']} | "
            f"{baseline['truth_recall']} | {baseline['false_candidate_count']} | "
            f"{v3['candidate_count']} | {v3['truth_recall']} | {v3['false_candidate_count']} | "
            f"{len(item['removed_true_candidate_ids'])} | "
            f"{len(item['removed_false_candidate_ids'])} | {item['timings_ms']['v3_filter']} |"
        )
    (output_dir / "comparison-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", metavar="LABEL=PATH")
    parser.add_argument("--truth-regions", required=True, type=Path)
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=SCRIPT_DIR / "cv_cross_experiment_config.json",
    )
    parser.add_argument(
        "--v3-config",
        type=Path,
        default=SCRIPT_DIR / "cv_cross_v3_experiment_config.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    baseline_config = _read_json(args.baseline_config)
    v3_config = _read_json(args.v3_config)
    truth_pages = _truth_pages(_read_json(args.truth_regions))
    summaries = []
    for value in args.images:
        label, separator, raw_path = value.partition("=")
        if not separator or label not in truth_pages:
            raise ValueError(f"invalid image argument: {value}")
        summaries.append(
            run_case(
                label,
                Path(raw_path),
                args.output,
                baseline_config,
                v3_config,
                truth_pages[label]["regions"],
            )
        )
    _write_json(args.output / "effective-baseline-config.json", baseline_config)
    _write_json(args.output / "effective-v3-config.json", v3_config)
    _write_json(args.output / "summary.json", summaries)
    write_report(args.output, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
