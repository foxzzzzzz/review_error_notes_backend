"""Offline convergence of two or more global semantic diagnostic runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from global_question_unit_convergence import (
    build_boundary_metadata,
    consolidate_retry_events,
)
from global_question_units import load_config


DEFAULT_CONFIG_PATH = Path(__file__).with_name("global_question_unit_config.json")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _fallback_events(page: Path, source_round: int) -> list[dict]:
    units = {
        str(item["question_unit_id"]): item
        for item in _load_json(page / "global-question-units.json")["units"]
    }
    guarded = _load_json(page / "semantic-geometry-guard-sets.json")
    audit = _load_json(page / "semantic-judge-audit.json")
    selected_ids = set(guarded["recall_safe_unit_ids"])
    anchors_by_unit = {unit_id: set() for unit_id in selected_ids}
    fits_by_unit = {unit_id: set() for unit_id in selected_ids}
    overrides = {
        int(item["cross_id"]): str(item["guarded_unit_id"])
        for item in guarded.get("overrides", [])
    }
    for decision in audit["accepted"]:
        cross_id = int(decision["cross_id"])
        unit_id = overrides.get(cross_id, decision.get("selected_unit_id"))
        if unit_id in selected_ids:
            anchors_by_unit[unit_id].add(cross_id)
            fits_by_unit[unit_id].add(str(decision.get("boundary_fit", "uncertain")))
    for item in guarded["needs_review"]:
        unit_id = item.get("fallback_unit_id")
        if unit_id in selected_ids:
            anchors_by_unit[unit_id].add(int(item["cross_id"]))
            fits_by_unit[unit_id].add("uncertain")
    return [
        {
            "question_unit_id": unit_id,
            "unit_bbox": list(units[unit_id]["unit_bbox"]),
            "anchor_ids": sorted(anchors_by_unit[unit_id]),
            "ocr_tokens": list(
                units[unit_id].get(
                    "ocr_tokens",
                    [f"line:{value}" for value in units[unit_id].get("ocr_line_ids", [])],
                )
            ),
            "source_round": source_round,
            "boundary_fits": sorted(fits_by_unit[unit_id]),
        }
        for unit_id in sorted(selected_ids)
        if unit_id in units
    ]


def _load_page_run(root: Path, label: str, source_round: int) -> dict:
    page = root / label
    units = _load_json(page / "global-question-units.json")["units"]
    for unit in units:
        unit.setdefault(
            "ocr_tokens",
            [f"line:{value}" for value in unit.get("ocr_line_ids", [])],
        )
    events_path = page / "semantic-recall-safe-events.json"
    if events_path.is_file():
        events = _load_json(events_path)
        for event in events:
            event["source_round"] = source_round
    else:
        events = _fallback_events(page, source_round)
    comparison = _load_json(page / "semantic-geometry-guard-safe-comparison.json")
    return {
        "units": units,
        "events": events,
        "anchor_candidates": _load_json(page / "anchor-unit-candidates.json")[
            "anchor_candidates"
        ],
        "comparison": comparison,
    }


def _audit_events(events: list[dict], run_pages: list[dict]) -> dict:
    truth_ids = set()
    truth_by_unit = {}
    intrusion_ids = set()
    for page in run_pages:
        comparison = page["comparison"]
        truth_ids.update(comparison.get("matched_truth_ids", []))
        truth_ids.update(comparison.get("missed_truth_ids", []))
        intrusion_ids.update(comparison.get("sibling_intrusion_unit_ids", []))
        for unit_id, matches in comparison.get("unit_truth_matches", {}).items():
            truth_by_unit.setdefault(unit_id, set()).update(matches)
    matches_by_truth = {truth_id: [] for truth_id in truth_ids}
    false_event_ids = []
    sibling_event_ids = []
    cross_truth_merge_conflict_ids = []
    for event_id, event in enumerate(events):
        source_matches = [
            truth_by_unit.get(unit_id, set())
            for unit_id in event["source_unit_ids"]
            if truth_by_unit.get(unit_id, set())
        ]
        matches = set().union(*source_matches) if source_matches else set()
        if (
            len(source_matches) > 1
            and len(matches) > 1
            and not set.intersection(*source_matches)
        ):
            cross_truth_merge_conflict_ids.append(event_id)
        if not matches:
            false_event_ids.append(event_id)
        for truth_id in matches:
            matches_by_truth.setdefault(truth_id, []).append(event_id)
        if event["question_unit_id"] in intrusion_ids:
            sibling_event_ids.append(event_id)
    matched = sorted(key for key, value in matches_by_truth.items() if value)
    return {
        "truth_count": len(truth_ids),
        "matched_truth_ids": matched,
        "missed_truth_ids": sorted(truth_ids - set(matched)),
        "truth_recall": round(len(matched) / len(truth_ids), 6) if truth_ids else 1.0,
        "false_event_ids": false_event_ids,
        "duplicate_truth_ids": sorted(
            key for key, value in matches_by_truth.items() if len(value) > 1
        ),
        "sibling_intrusion_event_ids": sibling_event_ids,
        "cross_truth_merge_conflict_event_ids": cross_truth_merge_conflict_ids,
    }


def _write_report(path: Path, summaries: list[dict]) -> None:
    lines = [
        "# 两轮识别本地收敛报告",
        "",
        "> 只回放已有结果；OCR＋空间去重和边界元数据均不增加LLM请求。",
        "",
        "| 图片 | 输入事件 | 稳定ID并集 | 稳定ID误报 | OCR空间合并 | 输出事件 | 并集召回 | 收敛召回 | 收敛误报 | 重复真值 | 跨真值误合并 | 兄弟题侵入 | 侵入已标记 | 边界歧义 | 本地耗时(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {label} | {input_event_count} | {stable_id_union_event_count} | "
            "{stable_id_union_false_event_count} | "
            "{conservative_ocr_geometry_merge_count} | {event_count} | "
            "{stable_id_union_truth_recall} | {truth_recall} | "
            "{false_event_count} | {duplicate_truth_count} | "
            "{cross_truth_merge_conflict_count} | "
            "{sibling_intrusion_event_count} | {flagged_sibling_intrusion_count} | "
            "{boundary_ambiguous_count} | {convergence_ms} |".format(
                **item
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(arguments=None) -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(arguments)
    if len(args.runs) < 2:
        parser.error("at least two runs are required")
    roots = [path.expanduser().resolve() for path in args.runs]
    labels = _load_json(roots[0] / "summary.json")["labels"]
    if any(_load_json(root / "summary.json")["labels"] != labels for root in roots[1:]):
        parser.error("all runs must contain the same labels in the same order")
    config = load_config(args.config.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for label in labels:
        page_started = time.perf_counter()
        run_pages = [
            _load_page_run(root, label, index)
            for index, root in enumerate(roots, start=1)
        ]
        convergence = consolidate_retry_events(
            [page["events"] for page in run_pages], config
        )
        units_by_id = {}
        candidates = {}
        for page in run_pages:
            units_by_id.update(
                {str(item["question_unit_id"]): item for item in page["units"]}
            )
            for cross_id, values in page["anchor_candidates"].items():
                known = {item["question_unit_id"] for item in candidates.get(cross_id, [])}
                candidates.setdefault(cross_id, []).extend(
                    item for item in values if item["question_unit_id"] not in known
                )
        boundary = build_boundary_metadata(
            convergence["events"], units_by_id, candidates, config
        )
        stable_id_audit = _audit_events(
            convergence["stable_id_union_events"], run_pages
        )
        audit = _audit_events(convergence["events"], run_pages)
        ambiguous_event_ids = {
            event_id
            for event_id, item in enumerate(boundary)
            if item["boundary_status"] == "boundary_ambiguous"
        }
        page_dir = output / label
        page_dir.mkdir()
        _write_json(page_dir / "retry-union-events.json", convergence["events"])
        _write_json(
            page_dir / "stable-id-union-events.json",
            convergence["stable_id_union_events"],
        )
        _write_json(page_dir / "retry-union-audit.json", convergence["audit"])
        _write_json(page_dir / "boundary-interaction.json", boundary)
        _write_json(page_dir / "truth-audit.json", audit)
        summary = {
            "label": label,
            "event_count": len(convergence["events"]),
            "stable_id_union_event_count": len(
                convergence["stable_id_union_events"]
            ),
            "input_event_count": convergence["audit"]["input_event_count"],
            "exact_id_merge_count": convergence["audit"]["exact_id_merge_count"],
            "conservative_ocr_geometry_merge_count": convergence["audit"][
                "conservative_ocr_geometry_merge_count"
            ],
            "truth_recall": audit["truth_recall"],
            "stable_id_union_truth_recall": stable_id_audit["truth_recall"],
            "stable_id_union_false_event_count": len(
                stable_id_audit["false_event_ids"]
            ),
            "false_event_count": len(audit["false_event_ids"]),
            "duplicate_truth_count": len(audit["duplicate_truth_ids"]),
            "cross_truth_merge_conflict_count": len(
                audit["cross_truth_merge_conflict_event_ids"]
            ),
            "sibling_intrusion_event_count": len(
                audit["sibling_intrusion_event_ids"]
            ),
            "flagged_sibling_intrusion_count": len(
                ambiguous_event_ids & set(audit["sibling_intrusion_event_ids"])
            ),
            "boundary_ambiguous_count": sum(
                item["boundary_status"] == "boundary_ambiguous"
                for item in boundary
            ),
            "convergence_ms": round(
                (time.perf_counter() - page_started) * 1000, 2
            ),
        }
        summaries.append(summary)
    aggregate = {
        "experiment": "global_question_unit_retry_convergence",
        "production_code_unchanged": True,
        "llm_request_count": 0,
        "labels": labels,
        "run_count": len(roots),
        "page_summaries": summaries,
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    _write_json(output / "summary.json", aggregate)
    _write_report(output / "comparison-report.md", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
