import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
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
    assert (
        "| page34 | 1 | 12 | 4 | None | None | None | None | 未运行 | "
        "None | None | None | None | None | None |"
    ) in report


def test_worker_mounts_diagnostic_script_directory_read_only():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    worker = compose.split("  worker:", 1)[1].split("  beat:", 1)[0]

    assert "./scripts:/app/scripts:ro" in worker


def test_stable_event_assignment_rejects_duplicate_primitive_membership():
    diagnostic = _load_script_module()
    with pytest.raises(ValueError, match="assigned more than once"):
        diagnostic.StableEventResult.model_validate(
            {
                "events": [
                    {
                        "event_id": 0,
                        "primitive_ids": [0, 1],
                        "event_type": "cross_circle",
                        "bbox": [0.1, 0.1, 0.4, 0.4],
                        "cross_bbox": [0.25, 0.1, 0.4, 0.25],
                        "circle_bbox": [0.1, 0.2, 0.35, 0.4],
                        "confidence": 0.9,
                    },
                    {
                        "event_id": 1,
                        "primitive_ids": [1],
                        "event_type": "cross",
                        "bbox": [0.5, 0.1, 0.6, 0.2],
                        "cross_bbox": [0.5, 0.1, 0.6, 0.2],
                        "circle_bbox": None,
                        "confidence": 0.8,
                    },
                ],
                "unassigned_primitive_ids": [],
            }
        )


def test_independent_mark_prompt_forbids_external_cv_coordinates_and_fragments():
    diagnostic = _load_script_module()

    prompt = diagnostic.INDEPENDENT_COMPLETE_MARK_PROMPT

    assert "不接收也不得推测任何外部候选坐标" in prompt
    assert "不得把圆弧、叉的一条笔画或孤立红线作为独立标记" in prompt
    assert "local_red_regions" not in prompt


def test_cv_audit_does_not_create_or_remove_stable_events():
    diagnostic = _load_script_module()
    result = diagnostic.StableEventResult.model_validate(
        {
            "events": [
                {
                    "event_id": 0,
                    "primitive_ids": [0, 1],
                    "event_type": "cross_circle",
                    "bbox": [0.1, 0.1, 0.5, 0.5],
                    "cross_bbox": [0.3, 0.1, 0.5, 0.3],
                    "circle_bbox": [0.1, 0.2, 0.4, 0.5],
                    "confidence": 0.9,
                }
            ],
            "unassigned_primitive_ids": [],
        }
    )
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:45, 15:40] = True
    components = [
        {"region_id": 0, "bbox": [0.15, 0.2, 0.4, 0.45], "pixel_count": 625},
        {"region_id": 1, "bbox": [0.7, 0.7, 0.8, 0.8], "pixel_count": 100},
    ]

    audit = diagnostic.audit_stable_events_against_cv(
        result,
        mask,
        components,
        min_red_pixels=12,
        min_red_ratio=0.01,
    )

    assert len(result.events) == 1
    assert audit["event_count_before"] == 1
    assert audit["event_count_after"] == 1
    assert audit["events"][0]["event_id"] == 0
    assert audit["events"][0]["status"] == "supported"
    assert audit["events"][0]["matched_component_ids"] == [0]
    assert audit["uncovered_component_ids"] == [1]


def test_duplicate_event_audit_reports_overlapping_separate_events():
    diagnostic = _load_script_module()
    result = diagnostic.StableEventResult.model_validate(
        {
            "events": [
                {
                    "event_id": 0,
                    "primitive_ids": [0],
                    "event_type": "circle",
                    "bbox": [0.1, 0.1, 0.5, 0.5],
                    "cross_bbox": None,
                    "circle_bbox": [0.1, 0.1, 0.5, 0.5],
                    "confidence": 0.9,
                },
                {
                    "event_id": 1,
                    "primitive_ids": [1],
                    "event_type": "cross",
                    "bbox": [0.2, 0.2, 0.45, 0.45],
                    "cross_bbox": [0.2, 0.2, 0.45, 0.45],
                    "circle_bbox": None,
                    "confidence": 0.8,
                },
            ],
            "unassigned_primitive_ids": [],
        }
    )

    candidates = diagnostic.find_duplicate_event_candidates(
        result,
        containment_threshold=0.8,
    )

    assert candidates == [
        {
            "event_ids": [0, 1],
            "intersection_over_smaller_area": 1.0,
        }
    ]


def test_comparison_summary_keeps_baseline_and_stable_event_counts_separate():
    diagnostic = _load_script_module()
    summary = diagnostic.build_summary(
        label="page34",
        expected_count=1,
        cv={"raw_component_count": 12, "evidence_group_count": 4},
        pipeline={
            "mark_primitive_count": 6,
            "mark_event_count": 3,
            "localized_mark_count": 3,
            "content_item_count": 3,
        },
        stable_experiment={
            "primitive_count": 2,
            "event_count": 1,
            "duplicate_primitive_ids": [],
            "unassigned_primitive_ids": [],
            "uncovered_component_ids": [2],
            "mark_detection_ms": 1000.0,
            "event_consolidation_ms": 700.0,
        },
    )

    checkpoints = summary["checkpoints"]
    assert checkpoints["normalized_mark_event_count"] == 3
    assert checkpoints["stable_event_count"] == 1
    assert checkpoints["stable_duplicate_primitive_count"] == 0
    assert checkpoints["stable_duplicate_event_candidate_count"] == 0
    assert checkpoints["stable_uncovered_component_count"] == 1


def test_stable_event_experiment_runs_independent_detection_before_cv_audit(tmp_path):
    diagnostic = _load_script_module()
    source = tmp_path / "page.jpg"
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((10, 20, 50, 50), fill=(220, 30, 30))
    image.save(source)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    calls = []

    class FakeClient:
        def detect_independent_complete_marks(self, image_path):
            calls.append(("detect", image_path))
            return diagnostic.MarkDetectionResult.model_validate(
                {
                    "error_marks": [
                        {
                            "mark_id": 0,
                            "mark_type": "circle",
                            "bbox": [0.1, 0.2, 0.5, 0.5],
                            "cross_bbox": None,
                            "circle_bbox": None,
                            "confidence": 0.9,
                        }
                    ]
                }
            )

        def consolidate_stable_events(self, image_path, primitives):
            calls.append(("consolidate", image_path, len(primitives)))
            return diagnostic.StableEventResult.model_validate(
                {
                    "events": [
                        {
                            "event_id": 0,
                            "primitive_ids": [0],
                            "event_type": "circle",
                            "bbox": [0.1, 0.2, 0.5, 0.5],
                            "cross_bbox": None,
                            "circle_bbox": [0.1, 0.2, 0.5, 0.5],
                            "confidence": 0.9,
                        }
                    ],
                    "unassigned_primitive_ids": [],
                }
            )

    result = diagnostic.run_stable_event_experiment(
        image_path=source,
        case_dir=case_dir,
        client=FakeClient(),
        cv={
            "red_mask": diagnostic._red_mask_for_image(source, 100),
            "components": [
                {
                    "region_id": 0,
                    "bbox": [0.1, 0.2, 0.5, 0.5],
                    "pixel_count": 1271,
                }
            ],
        },
    )

    assert [call[0] for call in calls] == ["detect", "consolidate"]
    assert result["event_count"] == 1
    assert result["cv_event_support"][0]["status"] == "supported"
    assert (case_dir / "stable-event-experiment" / "stable-events.json").is_file()
    assert (
        case_dir / "stable-event-experiment" / "cv-post-validation.json"
    ).is_file()
