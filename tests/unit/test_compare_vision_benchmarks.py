import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compare_vision_benchmarks.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_vision_benchmarks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(
    tmp_path,
    name,
    *,
    solution_id,
    truth_hash="truth-a",
    recall=True,
    false_count=0,
    timing_ms=1000.0,
    request_count=4,
):
    run_dir = tmp_path / name
    run_dir.mkdir()
    summary = {
        "schema_version": 1,
        "solution_id": solution_id,
        "truth_sha256": truth_hash,
        "image_sha256_by_label": {"page5": "image-a", "page7": "image-b"},
        "all_truth_recalled": recall,
        "truth_count": 2,
        "matched_truth_count": 2 if recall else 1,
        "truth_recall": 1.0 if recall else 0.5,
        "false_prediction_count": false_count,
        "llm_request_count": request_count,
        "core_timing_ms": timing_ms,
        "pages": [
            {
                "label": "page5",
                "truth_audit": {
                    "matched_truth_ids": ["page5-T1"] if recall else [],
                    "missed_truth_ids": [] if recall else ["page5-T1"],
                },
            }
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_rejects_runs_with_different_truth_hashes(tmp_path):
    compare = _load_module()
    first = _write_run(tmp_path, "first", solution_id="old_solution")
    second = _write_run(
        tmp_path,
        "second",
        solution_id="main_single_pass",
        truth_hash="truth-b",
    )

    with pytest.raises(ValueError, match="truth hash"):
        compare.load_compatible_runs([first, second])


def test_aggregates_recall_false_regions_requests_and_timing(tmp_path):
    compare = _load_module()
    runs = compare.load_compatible_runs(
        [
            _write_run(
                tmp_path,
                "old-1",
                solution_id="old_solution",
                false_count=1,
                timing_ms=1000.0,
                request_count=2,
            ),
            _write_run(
                tmp_path,
                "old-2",
                solution_id="old_solution",
                false_count=3,
                timing_ms=1200.0,
                request_count=2,
            ),
            _write_run(
                tmp_path,
                "main-1",
                solution_id="main_single_pass",
                recall=True,
                timing_ms=800.0,
                request_count=5,
            ),
            _write_run(
                tmp_path,
                "main-2",
                solution_id="main_single_pass",
                recall=False,
                timing_ms=900.0,
                request_count=4,
            ),
            _write_run(
                tmp_path,
                "main-3",
                solution_id="main_single_pass",
                recall=True,
                timing_ms=1000.0,
                request_count=5,
            ),
        ]
    )

    report = compare.aggregate_runs(runs)

    assert report["old_solution"]["core_timing_ms"] == {
        "mean": 1100.0,
        "p50": 1100.0,
        "worst": 1200.0,
    }
    assert report["old_solution"]["false_prediction_count_mean"] == 2
    assert report["main_single_pass"]["all_truth_recall_success_rate"] == pytest.approx(2 / 3)
    assert report["main_single_pass"]["llm_request_count_mean"] == 4.667
