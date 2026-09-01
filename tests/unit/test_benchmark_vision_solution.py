import importlib.util
import json
from pathlib import Path

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_vision_solution.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_vision_solution", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalizes_single_pass_stable_events_to_common_predictions():
    benchmark = _load_module()
    stable_events = {
        "run_count": 1,
        "events": [
            {
                "event_id": 3,
                "cross_ids": [7],
                "question_bboxes": [[0.1, 0.2, 0.4, 0.5]],
                "confidence": 0.91,
            }
        ],
    }
    experiment_summary = {
        "llm2_localization_run_count": 1,
        "llm_request_count": 4,
        "timings_ms": {"total": 1250.0, "post_llm2_audit": 10.0},
    }

    result = benchmark.normalize_main_page_result(
        stable_events=stable_events,
        experiment_summary=experiment_summary,
        cross_cv_timing_ms=20.0,
    )

    assert result["solution_id"] == "main_single_pass"
    assert result["llm_request_count"] == 4
    assert result["predictions"] == [
        {
            "prediction_id": "P1",
            "event_id": 3,
            "cross_ids": [7],
            "bbox": [0.1, 0.2, 0.4, 0.5],
            "confidence": 0.91,
        }
    ]
    assert result["timing_ms"]["core_total"] == 1260.0
    assert result["timing_ms"]["ocr"] == 0


def test_rejects_normalization_when_second_llm2_pass_ran():
    benchmark = _load_module()

    try:
        benchmark.normalize_main_page_result(
            stable_events={"run_count": 2, "events": []},
            experiment_summary={
                "llm2_localization_run_count": 2,
                "llm_request_count": 5,
                "timings_ms": {"total": 1500.0},
            },
            cross_cv_timing_ms=20.0,
        )
    except ValueError as exc:
        assert "single LLM2 pass" in str(exc)
    else:
        raise AssertionError("second-pass output must be rejected")


def test_parser_accepts_page33_to_35_and_new_pages(tmp_path):
    benchmark = _load_module()
    labels = ["page33", "page34", "page35", "page5", "page7", "page20"]
    image_arguments = []
    for label in labels:
        path = tmp_path / f"{label}.jpg"
        Image.new("RGB", (10, 10), "white").save(path)
        image_arguments.append(f"{label}={path}")
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "pages": {
                    label: {
                        "regions": [
                            {
                                "truth_id": f"{label}-T1",
                                "source_bbox_normalized": [0.1, 0.1, 0.3, 0.3],
                            }
                        ]
                    }
                    for label in labels
                }
            }
        ),
        encoding="utf-8",
    )

    args = benchmark.parse_args(
        image_arguments
        + [
            "--truth-regions",
            str(truth_path),
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert [label for label, _path in args.images] == labels
