import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw


BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "diagnose_vision_pipeline.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("diagnose_vision_pipeline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_cv_artifacts_records_components_groups_and_overlay(tmp_path):
    diagnostic = _load_script_module()
    source = tmp_path / "page.jpg"
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 35, 35), fill=(220, 30, 30))
    draw.rectangle((42, 20, 57, 35), fill=(220, 30, 30))
    image.save(source)

    output = tmp_path / "diagnostic"
    result = diagnostic.write_cv_artifacts(
        source,
        output,
        max_edge=200,
        min_component_pixels=4,
        max_component_area_ratio=0.5,
        max_thinness_ratio=20,
        group_max_gap_ratio=0.05,
        group_max_area_ratio=0.5,
    )

    assert result["raw_component_count"] == 2
    assert result["evidence_group_count"] == 1
    assert (output / "cv" / "components-and-groups.jpg").is_file()
    assert (output / "cv" / "red-mask.png").is_file()
    payload = json.loads((output / "cv" / "evidence.json").read_text("utf-8"))
    assert len(payload["components"]) == 2
    assert len(payload["groups"]) == 1


def test_exchange_recorder_writes_each_llm_event_inside_current_call(tmp_path):
    diagnostic = _load_script_module()
    recorder = diagnostic.ExchangeRecorder(tmp_path)
    call_dir = recorder.begin_call("mark_detection", {"correction": None})

    recorder(
        {
            "kind": "parsed_response",
            "operation": "mark_detection",
            "attempt": 1,
            "raw": {"error_marks": []},
        }
    )
    recorder.finish_call(call_dir, result={"error_marks": []})

    assert call_dir.name == "call-001-mark_detection"
    event = json.loads((call_dir / "event-001-parsed_response.json").read_text("utf-8"))
    assert event["raw"] == {"error_marks": []}
    assert json.loads((call_dir / "result.json").read_text("utf-8")) == {
        "error_marks": []
    }


def test_build_summary_identifies_first_count_divergence_without_claiming_cv_failure():
    diagnostic = _load_script_module()

    summary = diagnostic.build_summary(
        label="page34",
        expected_count=1,
        cv={"raw_component_count": 12, "evidence_group_count": 4},
        pipeline={
            "mark_primitive_count": 4,
            "mark_event_count": 4,
            "localized_mark_count": 4,
            "content_item_count": 4,
        },
    )

    assert summary["first_count_divergence"] == "mark_detection_or_grouping"
    assert summary["checkpoints"]["cv_evidence_group_count"] == 4
    assert "cv_failure" not in json.dumps(summary)


def test_build_summary_does_not_infer_divergence_for_unrun_llm_stages():
    diagnostic = _load_script_module()

    summary = diagnostic.build_summary(
        label="page34",
        expected_count=1,
        cv={"raw_component_count": 12, "evidence_group_count": 4},
        pipeline=None,
    )

    assert summary["first_count_divergence"] is None
    assert summary["pipeline_status"] == "not_run"


def test_report_labels_unrun_pipeline_instead_of_claiming_no_divergence(tmp_path):
    diagnostic = _load_script_module()
    summary = diagnostic.build_summary(
        label="page34",
        expected_count=1,
        cv={"raw_component_count": 12, "evidence_group_count": 4},
        pipeline=None,
    )

    diagnostic._write_report(tmp_path, [summary])

    report = (tmp_path / "comparison-report.md").read_text("utf-8")
    assert "| page34 | 1 | 12 | 4 | None | None | None | 未运行 |" in report


def test_worker_mounts_diagnostic_script_directory_read_only():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    worker = compose.split("  worker:", 1)[1].split("  beat:", 1)[0]

    assert "./scripts:/app/scripts:ro" in worker
