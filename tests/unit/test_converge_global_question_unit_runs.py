import importlib.util
import json
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "converge_global_question_unit_runs.py"


def _load_script():
    scripts_path = str(BACKEND_ROOT / "scripts")
    import sys

    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "converge_global_question_unit_runs", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run_fixture(root, unit_id, bbox, *, round_selected=True):
    _write_json(root / "summary.json", {"labels": ["page33"]})
    page = root / "page33"
    _write_json(
        page / "global-question-units.json",
        {
            "units": [
                {
                    "question_unit_id": unit_id,
                    "unit_bbox": bbox,
                    "ocr_line_ids": [7],
                    "ocr_tokens": ["same"],
                }
            ]
        },
    )
    _write_json(
        page / "anchor-unit-candidates.json",
        {
            "anchor_candidates": {
                "1": [
                    {
                        "question_unit_id": unit_id,
                        "rank": 1,
                        "anchor_in_unit": True,
                    }
                ]
            }
        },
    )
    _write_json(
        page / "semantic-geometry-guard-sets.json",
        {
            "recall_safe_unit_ids": [unit_id] if round_selected else [],
            "needs_review": [],
            "overrides": [],
        },
    )
    _write_json(
        page / "semantic-judge-audit.json",
        {
            "accepted": [
                {
                    "cross_id": 1,
                    "selected_unit_id": unit_id,
                    "boundary_fit": "complete",
                }
            ],
            "accepted_supplemental_unit_ids": [],
            "violations": [],
        },
    )
    _write_json(
        page / "semantic-geometry-guard-safe-comparison.json",
        {
            "matched_truth_ids": ["T1"],
            "missed_truth_ids": [],
            "false_unit_ids": [],
            "duplicate_truth_ids": [],
            "sibling_intrusion_unit_ids": [],
            "unit_truth_matches": {unit_id: ["T1"]},
        },
    )


def test_cli_conservatively_merges_two_archived_runs_without_llm(tmp_path):
    script = _load_script()
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_fixture(first, "wide", [0.1, 0.1, 0.5, 0.4])
    _run_fixture(second, "tight", [0.12, 0.11, 0.48, 0.38])
    output = tmp_path / "merged"

    assert script.main(
        [str(first), str(second), "--output", str(output)]
    ) == 0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    page = summary["page_summaries"][0]
    assert summary["llm_request_count"] == 0
    assert page["truth_recall"] == 1.0
    assert page["stable_id_union_truth_recall"] == 1.0
    assert page["stable_id_union_event_count"] == 2
    assert page["event_count"] == 1
    assert page["conservative_ocr_geometry_merge_count"] == 1
    assert page["cross_truth_merge_conflict_count"] == 0
    assert page["flagged_sibling_intrusion_count"] == 0
    assert page["convergence_ms"] >= 0
    assert summary["total_ms"] >= page["convergence_ms"]
    assert (output / "page33/retry-union-events.json").is_file()
    assert (output / "page33/boundary-interaction.json").is_file()
    assert (output / "comparison-report.md").is_file()
