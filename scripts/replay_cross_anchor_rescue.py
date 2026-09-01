"""Replay independent-scan anchor rescue from existing diagnostic archives."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_vision_pipeline import (
    CrossCandidateVerificationResult,
    IndependentCrossScanResult,
    _bbox_center_distance,
    _bbox_iou,
    _write_json,
    compare_cross_candidates_to_truth,
    select_independent_rescue_crosses,
)


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(root) or member.issym() or member.islnk():
            raise ValueError(f"unsafe tar member: {member.name}")
    archive.extractall(destination)


def _find_benchmark_root(path: Path) -> Path:
    if (path / "pages").is_dir():
        return path
    matches = sorted(
        candidate.parent
        for candidate in path.rglob("pages")
        if candidate.is_dir()
    )
    if len(matches) != 1:
        raise ValueError(f"expected one benchmark pages directory in {path}")
    return matches[0]


@contextmanager
def _open_archive(path: Path):
    if path.is_dir():
        yield _find_benchmark_root(path)
        return
    with tempfile.TemporaryDirectory(prefix="cross-anchor-replay-") as temp_dir:
        destination = Path(temp_dir)
        with tarfile.open(path, "r:gz") as archive:
            _safe_extract_tar(archive, destination)
        yield _find_benchmark_root(destination)


def _archive_label(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".tar.gz") else path.stem


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _find_source_image(page_dir: Path) -> Path:
    matches = sorted(page_dir.glob("source.*"))
    if len(matches) != 1:
        raise ValueError(f"expected one source image in {page_dir}")
    return matches[0]


def _union_anchors(
    anchors: list[dict],
    rescued: IndependentCrossScanResult,
    config: dict,
) -> list[dict]:
    union = [dict(anchor) for anchor in anchors]
    for cross in rescued.crosses:
        matching = next(
            (
                anchor
                for anchor in union
                if _bbox_iou(anchor["bbox"], cross.bbox)
                >= float(config["cross_anchor_fallback_merge_iou_threshold"])
                or _bbox_center_distance(anchor["bbox"], cross.bbox)
                <= float(
                    config["cross_anchor_fallback_merge_center_distance_ratio"]
                )
            ),
            None,
        )
        if matching is None:
            union.append(
                {
                    "source": "independent_scan_rescue",
                    "source_candidate_ids": [],
                    "bbox": list(cross.bbox),
                    "confidence": cross.confidence,
                    "independent_scan_supported": True,
                    "merge_reason": None,
                }
            )
        else:
            matching["independent_scan_supported"] = True
    union.sort(
        key=lambda item: (
            (item["bbox"][1] + item["bbox"][3]) / 2,
            (item["bbox"][0] + item["bbox"][2]) / 2,
            item["source"],
        )
    )
    for cross_id, anchor in enumerate(union):
        anchor["cross_id"] = cross_id
    return union


def _candidate_payload(anchors: list[dict]) -> list[dict]:
    return [
        {
            "candidate_id": anchor["cross_id"],
            "bbox": anchor["bbox"],
            "center": [
                round((anchor["bbox"][0] + anchor["bbox"][2]) / 2, 6),
                round((anchor["bbox"][1] + anchor["bbox"][3]) / 2, 6),
            ],
        }
        for anchor in anchors
    ]


def _replay_page(
    page_dir: Path,
    output_dir: Path,
    truth_regions: list[dict],
    config: dict,
) -> dict:
    cross_dir = page_dir / "cross-anchor-experiment"
    anchors = _read_json(cross_dir / "confirmed-crosses.json")
    independent_scan = IndependentCrossScanResult.model_validate(
        _read_json(cross_dir / "llm1-independent-scan.json")
    )
    fallback_verification = CrossCandidateVerificationResult.model_validate(
        _read_json(cross_dir / "llm1-fallback-candidate-verification.json")
    )
    rescued, audit = select_independent_rescue_crosses(
        existing_anchors=anchors,
        independent_scan=independent_scan,
        fallback_verification=fallback_verification,
        image_path=_find_source_image(page_dir),
        config=config,
    )
    union = _union_anchors(anchors, rescued, config)
    baseline_comparison = compare_cross_candidates_to_truth(
        _candidate_payload(anchors),
        truth_regions,
        margin_ratio=float(config.get("truth_match_margin_ratio", 0.0)),
    )
    comparison = compare_cross_candidates_to_truth(
        _candidate_payload(union),
        truth_regions,
        margin_ratio=float(config.get("truth_match_margin_ratio", 0.0)),
    )
    assignment_by_id = {
        item["candidate_id"]: item for item in comparison["assignments"]
    }
    rescued_ids = [
        anchor["cross_id"]
        for anchor in union
        if anchor["source"] == "independent_scan_rescue"
    ]
    added_false_ids = [
        cross_id
        for cross_id in rescued_ids
        if assignment_by_id[cross_id]["truth_id"] is None
    ]
    truth_assignment_counts = {}
    for assignment in comparison["assignments"]:
        truth_id = assignment["truth_id"]
        if truth_id is not None:
            truth_assignment_counts[truth_id] = truth_assignment_counts.get(truth_id, 0) + 1
    duplicate_candidate_count = sum(
        count - 1 for count in truth_assignment_counts.values() if count > 1
    )
    baseline_matched = set(baseline_comparison["matched_truth_ids"])
    recovered_truth_ids = [
        truth_id
        for truth_id in comparison["matched_truth_ids"]
        if truth_id not in baseline_matched
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "replay-anchor-union.json", union)
    _write_json(output_dir / "replay-anchor-rescue-audit.json", audit)
    _write_json(output_dir / "replay-anchor-truth-comparison.json", comparison)
    return {
        "page": page_dir.name,
        "baseline_anchor_count": len(anchors),
        "union_anchor_count": len(union),
        "rescued_anchor_count": len(rescued_ids),
        "added_false_anchor_count": len(added_false_ids),
        "added_false_anchor_ids": added_false_ids,
        "duplicate_candidate_count": duplicate_candidate_count,
        "matched_truth_count": comparison["matched_truth_count"],
        "truth_count": comparison["truth_count"],
        "truth_recall": comparison["truth_recall"],
        "matched_truth_ids": comparison["matched_truth_ids"],
        "recovered_truth_ids": recovered_truth_ids,
        "missed_truth_ids": comparison["missed_truth_ids"],
    }


def _write_report(output_dir: Path, summaries: list[dict]) -> None:
    lines = [
        "# 独立扫描补锚离线回放",
        "",
        "> truth 仅用于补锚完成后的审计，不参与候选决策。",
        "",
        "| 归档 | 页面 | 原锚点 | 补锚 | 新增假锚点 | 恢复 truth | 真值命中 | 真值召回 | 漏检 truth |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            "| {archive} | {page} | {baseline_anchor_count} | "
            "{rescued_anchor_count} | {added_false_anchor_count} | {recovered} | "
            "{matched_truth_count}/{truth_count} | {truth_recall} | {missed} |".format(
                **item,
                recovered=", ".join(item["recovered_truth_ids"]) or "-",
                missed=", ".join(item["missed_truth_ids"]) or "-",
            )
        )
    (output_dir / "replay-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--truth-regions", action="append", required=True)
    parser.add_argument("--cross-cv-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    truth_pages = {}
    for truth_value in args.truth_regions:
        for label, page in _read_json(Path(truth_value))["pages"].items():
            if label in truth_pages:
                raise ValueError(f"duplicate truth page: {label}")
            truth_pages[label] = page
    config = _read_json(Path(args.cross_cv_config))
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for archive_value in args.archive:
        archive_path = Path(archive_value).resolve()
        archive_label = _archive_label(archive_path)
        with _open_archive(archive_path) as root:
            for page_dir in sorted((root / "pages").iterdir()):
                if not page_dir.is_dir():
                    continue
                truth_page = truth_pages.get(page_dir.name)
                if not isinstance(truth_page, dict):
                    raise ValueError(f"missing truth page: {page_dir.name}")
                summary = _replay_page(
                    page_dir,
                    output_dir / archive_label / "pages" / page_dir.name,
                    truth_page["regions"],
                    config,
                )
                summary["archive"] = archive_label
                summaries.append(summary)
    _write_json(output_dir / "summary.json", summaries)
    _write_report(output_dir, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
