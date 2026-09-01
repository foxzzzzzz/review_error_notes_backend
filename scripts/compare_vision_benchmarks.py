"""Compare compatible old_solution and main_single_pass benchmark runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def load_compatible_runs(paths: list[Path]) -> list[dict]:
    if len(paths) < 2:
        raise ValueError("at least two benchmark run directories are required")
    runs = []
    for path in paths:
        summary_path = path / "summary.json"
        if not summary_path.is_file():
            raise ValueError(f"missing summary.json: {path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["run_path"] = str(path.resolve())
        runs.append(summary)

    schema_versions = {run.get("schema_version") for run in runs}
    if len(schema_versions) != 1:
        raise ValueError("schema version differs between benchmark runs")
    truth_hashes = {run.get("truth_sha256") for run in runs}
    if len(truth_hashes) != 1:
        raise ValueError("truth hash differs between benchmark runs")
    image_hashes = {
        tuple((run.get("image_sha256_by_label") or {}).items()) for run in runs
    }
    if len(image_hashes) != 1:
        raise ValueError("image hashes or label order differ between benchmark runs")
    return runs


def _round(value: float) -> float:
    return round(float(value), 3)


def aggregate_runs(runs: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for run in runs:
        solution_id = run.get("solution_id")
        if not isinstance(solution_id, str) or not solution_id:
            raise ValueError("benchmark run is missing solution_id")
        grouped[solution_id].append(run)

    aggregated = {}
    for solution_id, items in sorted(grouped.items()):
        timings = [float(item["core_timing_ms"]) for item in items]
        false_counts = [int(item["false_prediction_count"]) for item in items]
        request_counts = [int(item["llm_request_count"]) for item in items]
        truth_observations = defaultdict(list)
        for item in items:
            for page in item.get("pages", []):
                audit = page.get("truth_audit") or {}
                for truth_id in audit.get("matched_truth_ids", []):
                    truth_observations[(page["label"], truth_id)].append(True)
                for truth_id in audit.get("missed_truth_ids", []):
                    truth_observations[(page["label"], truth_id)].append(False)
        truth_success_rates = {
            f"{label}/{truth_id}": _round(sum(values) / len(values))
            for (label, truth_id), values in sorted(truth_observations.items())
        }
        aggregated[solution_id] = {
            "run_count": len(items),
            "all_truth_recall_success_rate": sum(
                bool(item["all_truth_recalled"]) for item in items
            )
            / len(items),
            "truth_recall_mean": _round(
                statistics.mean(float(item["truth_recall"]) for item in items)
            ),
            "false_prediction_count_mean": _round(statistics.mean(false_counts)),
            "llm_request_count_mean": _round(statistics.mean(request_counts)),
            "core_timing_ms": {
                "mean": _round(statistics.mean(timings)),
                "p50": _round(statistics.median(timings)),
                "worst": _round(max(timings)),
            },
            "truth_success_rates": truth_success_rates,
            "runs": [
                {
                    "run_path": item["run_path"],
                    "all_truth_recalled": item["all_truth_recalled"],
                    "truth_recall": item["truth_recall"],
                    "false_prediction_count": item["false_prediction_count"],
                    "llm_request_count": item["llm_request_count"],
                    "core_timing_ms": item["core_timing_ms"],
                }
                for item in items
            ],
        }
    return aggregated


def _write_report(path: Path, report: dict[str, dict]) -> None:
    lines = [
        "# 双方案视觉基准汇总",
        "",
        "| 方案 | 运行次数 | 全真值覆盖成功率 | 平均召回 | 平均误报 | 平均LLM请求 | 平均耗时(ms) | P50耗时(ms) | 最差耗时(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for solution_id, item in report.items():
        timing = item["core_timing_ms"]
        lines.append(
            f"| {solution_id} | {item['run_count']} | {item['all_truth_recall_success_rate']:.3f} | "
            f"{item['truth_recall_mean']} | {item['false_prediction_count_mean']} | "
            f"{item['llm_request_count_mean']} | {timing['mean']} | {timing['p50']} | {timing['worst']} |"
        )
    lines.extend(["", "## 每道真值多轮命中率", ""])
    truth_keys = sorted(
        {key for item in report.values() for key in item["truth_success_rates"]}
    )
    header = "| 真值 | " + " | ".join(report) + " |"
    separator = "|---|" + "---:|" * len(report)
    lines.extend([header, separator])
    for truth_key in truth_keys:
        values = [str(report[solution_id]["truth_success_rates"].get(truth_key)) for solution_id in report]
        lines.append(f"| {truth_key} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Extracted benchmark output directories")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = parse_args(arguments)
    runs = load_compatible_runs(args.runs)
    report = aggregate_runs(runs)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "solution-comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(args.output / "solution-comparison.md", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
