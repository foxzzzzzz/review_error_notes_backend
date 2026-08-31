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


def test_detect_red_cross_candidates_finds_faint_diagonal_cross_not_printed_grid(
    tmp_path,
):
    diagnostic = _load_script_module()
    source = tmp_path / "page.png"
    image = Image.new("RGB", (200, 140), "white")
    draw = ImageDraw.Draw(image)
    grid_color = (215, 150, 150)
    for x in (20, 60, 100, 140, 180):
        draw.line((x, 10, x, 130), fill=grid_color, width=2)
    for y in (20, 60, 100):
        draw.line((10, y, 190, y), fill=grid_color, width=2)
    faint_teacher_red = (170, 135, 125)
    draw.line((72, 42, 112, 82), fill=faint_teacher_red, width=4)
    draw.line((112, 42, 72, 82), fill=faint_teacher_red, width=4)
    image.save(source)

    result = diagnostic.detect_red_cross_candidates(
        source,
        {
            "analysis_max_edge": 400,
            "red_min_channel": 80,
            "red_min_excess": 12,
            "axis_line_min_density": 0.7,
            "arm_inner_radius_ratio": 0.015,
            "arm_outer_radius_ratio": 0.11,
            "diagonal_band_ratio": 0.2,
            "arm_min_density": 0.12,
            "center_radius_ratio": 0.015,
            "center_min_density": 0.08,
            "candidate_merge_radius_ratio": 0.06,
            "bbox_padding_ratio": 0.015,
        },
    )

    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["candidate_id"] == 0
    left, top, right, bottom = candidate["bbox"]
    assert left <= 0.36
    assert top <= 0.31
    assert right >= 0.56
    assert bottom >= 0.58
    assert candidate["center"] == pytest.approx([0.46, 0.45], abs=0.02)
    assert candidate["min_arm_density"] >= 0.12


def test_cross_candidate_truth_comparison_assigns_each_candidate_once():
    diagnostic = _load_script_module()
    candidates = [
        {"candidate_id": 0, "center": [0.2, 0.2], "bbox": [0.1, 0.1, 0.3, 0.3]},
        {"candidate_id": 1, "center": [0.7, 0.7], "bbox": [0.6, 0.6, 0.8, 0.8]},
        {"candidate_id": 2, "center": [0.95, 0.1], "bbox": [0.9, 0.05, 1.0, 0.15]},
    ]
    truth_regions = [
        {"truth_id": "T1", "source_bbox_normalized": [0.05, 0.05, 0.4, 0.4]},
        {"truth_id": "T2", "source_bbox_normalized": [0.55, 0.55, 0.85, 0.85]},
    ]

    comparison = diagnostic.compare_cross_candidates_to_truth(
        candidates,
        truth_regions,
    )

    assert comparison == {
        "truth_count": 2,
        "candidate_count": 3,
        "matched_truth_count": 2,
        "truth_recall": 1.0,
        "matched_truth_ids": ["T1", "T2"],
        "missed_truth_ids": [],
        "false_candidate_ids": [2],
        "assignments": [
            {"candidate_id": 0, "truth_id": "T1"},
            {"candidate_id": 1, "truth_id": "T2"},
            {"candidate_id": 2, "truth_id": None},
        ],
    }


def test_cross_candidate_truth_comparison_allows_configured_boundary_margin():
    diagnostic = _load_script_module()

    comparison = diagnostic.compare_cross_candidates_to_truth(
        [{"candidate_id": 0, "center": [0.49, 0.49], "bbox": [0.45, 0.45, 0.55, 0.55]}],
        [{"truth_id": "T1", "source_bbox_normalized": [0.5, 0.5, 0.8, 0.8]}],
        margin_ratio=0.02,
    )

    assert comparison["matched_truth_ids"] == ["T1"]
    assert comparison["false_candidate_ids"] == []


def test_red_cross_detection_is_deterministic_for_same_image(tmp_path):
    diagnostic = _load_script_module()
    source = tmp_path / "cross.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((25, 25, 75, 75), fill=(190, 80, 80), width=5)
    draw.line((75, 25, 25, 75), fill=(190, 80, 80), width=5)
    image.save(source)
    config = {
        "analysis_max_edge": 200,
        "red_min_channel": 80,
        "red_min_excess": 12,
        "axis_line_min_density": 0.7,
        "arm_inner_radius_ratio": 0.02,
        "arm_outer_radius_ratio": 0.15,
        "diagonal_band_ratio": 0.2,
        "arm_min_density": 0.12,
        "center_radius_ratio": 0.02,
        "center_min_density": 0.08,
        "candidate_merge_radius_ratio": 0.08,
        "bbox_padding_ratio": 0.02,
    }

    first = diagnostic.detect_red_cross_candidates(source, config)
    second = diagnostic.detect_red_cross_candidates(source, config)

    assert first["candidates"] == second["candidates"]
    assert np.array_equal(first["red_mask"], second["red_mask"])
    assert np.array_equal(first["candidate_center_mask"], second["candidate_center_mask"])


def test_write_cross_cv_artifacts_compares_candidates_with_truth(tmp_path):
    diagnostic = _load_script_module()
    source = tmp_path / "cross.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((25, 25, 75, 75), fill=(190, 80, 80), width=5)
    draw.line((75, 25, 25, 75), fill=(190, 80, 80), width=5)
    image.save(source)
    output = tmp_path / "case"
    output.mkdir()

    result = diagnostic.write_cross_cv_artifacts(
        source,
        output,
        {
            "analysis_max_edge": 200,
            "red_min_channel": 80,
            "red_min_excess": 12,
            "axis_line_min_density": 0.7,
            "arm_inner_radius_ratio": 0.02,
            "arm_outer_radius_ratio": 0.15,
            "diagonal_band_ratio": 0.2,
            "arm_min_density": 0.12,
            "center_radius_ratio": 0.02,
            "center_min_density": 0.08,
            "candidate_merge_radius_ratio": 0.08,
            "bbox_padding_ratio": 0.02,
        },
        [
            {
                "truth_id": "T1",
                "source_bbox_normalized": [0.15, 0.15, 0.85, 0.85],
            }
        ],
    )

    assert result["candidate_count"] == 1
    assert result["truth_comparison"]["truth_recall"] == 1.0
    artifact_dir = output / "cv-cross-experiment"
    assert (artifact_dir / "candidates.json").is_file()
    assert (artifact_dir / "truth-comparison.json").is_file()
    assert (artifact_dir / "red-mask.png").is_file()
    assert (artifact_dir / "geometry-mask.png").is_file()
    assert (artifact_dir / "candidate-centers.png").is_file()
    assert (artifact_dir / "candidates-overlay.jpg").is_file()
    assert (artifact_dir / "truth-candidates-overlay.jpg").is_file()


def test_write_cross_candidate_montage_keeps_full_page_and_enlarged_tiles(tmp_path):
    diagnostic = _load_script_module()
    source = tmp_path / "page.jpg"
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line((30, 30, 50, 50), fill="red", width=3)
    draw.line((50, 30, 30, 50), fill="red", width=3)
    image.save(source)
    output = tmp_path / "montage.jpg"

    result = diagnostic.write_cross_candidate_montage(
        source,
        output,
        [
            {
                "candidate_id": 7,
                "bbox": [0.1, 0.1, 0.3, 0.5],
                "center": [0.2, 0.3],
            }
        ],
        full_page_max_edge=200,
        tile_edge=120,
        columns=2,
        crop_padding_ratio=0.03,
    )

    assert result == {"candidate_count": 1, "tile_candidate_ids": [7]}
    assert output.is_file()
    with Image.open(output) as montage:
        assert montage.width > image.width
        assert montage.height >= image.height


def test_run_case_cv_only_includes_cross_truth_summary_without_llm(tmp_path):
    diagnostic = _load_script_module()
    source = tmp_path / "cross.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((25, 25, 75, 75), fill=(190, 80, 80), width=5)
    draw.line((75, 25, 25, 75), fill=(190, 80, 80), width=5)
    image.save(source)
    config = {
        "analysis_max_edge": 200,
        "red_min_channel": 80,
        "red_min_excess": 12,
        "axis_line_min_density": 0.7,
        "arm_inner_radius_ratio": 0.02,
        "arm_outer_radius_ratio": 0.15,
        "diagonal_band_ratio": 0.2,
        "arm_min_density": 0.12,
        "center_radius_ratio": 0.02,
        "center_min_density": 0.08,
        "candidate_merge_radius_ratio": 0.08,
        "bbox_padding_ratio": 0.02,
    }

    summary = diagnostic.run_case(
        label="sample",
        image_path=source,
        output_dir=tmp_path / "output",
        expected_count=1,
        subject_hint="chinese",
        cv_only=True,
        cross_cv_config=config,
        truth_regions=[
            {
                "truth_id": "T1",
                "source_bbox_normalized": [0.15, 0.15, 0.85, 0.85],
            }
        ],
    )

    assert summary["pipeline_status"] == "not_run"
    assert summary["cv_cross_experiment"]["candidate_count"] == 1
    assert summary["cv_cross_experiment"]["truth_comparison"]["truth_recall"] == 1.0


def test_load_cross_cv_inputs_requires_config_and_truth_for_every_label(tmp_path):
    diagnostic = _load_script_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "analysis_max_edge": 1600,
                "red_min_channel": 70,
                "red_min_excess": 8,
                "axis_line_min_density": 0.7,
                "arm_inner_radius_ratio": 0.004,
                "arm_outer_radius_ratio": 0.025,
                "diagonal_band_ratio": 0.25,
                "arm_min_density": 0.12,
                "center_radius_ratio": 0.003,
                "center_min_density": 0.08,
                "candidate_merge_radius_ratio": 0.02,
                "bbox_padding_ratio": 0.005,
                "truth_match_margin_ratio": 0.01,
                "cross_anchor_question_max_area_ratio": 0.3,
                "cross_anchor_question_max_gap_ratio": 0.04,
                "cross_anchor_duplicate_question_iou_threshold": 0.2,
                "cross_anchor_llm1_verification_runs": 1,
                "cross_anchor_llm2_localization_runs": 2,
                "cross_anchor_llm2_retry_min_center_gap_ratio": 0.03,
                "cross_anchor_llm2_retry_crop_padding_ratio": 0.12,
                "cross_anchor_llm2_retry_first_pass_min_question_cross_area_ratio": 1.1,
                "cross_anchor_llm2_retry_min_question_cross_area_ratio": 1.5,
                "cross_anchor_llm2_retry_max_question_cross_iou": 0.8,
                "cross_anchor_llm2_retry_shared_question_iou_threshold": 0.8,
                "cross_anchor_llm2_retry_shared_question_min_anchor_distance_ratio": 0.1,
                "cross_anchor_llm2_retry_shared_question_min_group_size": 3,
                "cross_anchor_llm2_retry_max_requests_per_page": 1,
                "cross_anchor_fallback_generates_anchors": False,
                "cross_anchor_retain_uncertain_candidates": True,
                "cross_anchor_retain_rejected_candidates": True,
                "cross_anchor_retain_uncertain_fallback_candidates": True,
                "cross_anchor_high_cv_min_arm_density": 0.25,
                "cross_anchor_high_cv_min_center_density": 0.7,
                "cross_anchor_cv_dedupe_iou_threshold": 0.5,
                "cross_anchor_cv_dedupe_center_distance_ratio": 0.015,
                "cross_anchor_fallback_merge_iou_threshold": 0.2,
                "cross_anchor_fallback_merge_center_distance_ratio": 0.04,
                "cross_anchor_montage_full_page_max_edge": 1400,
                "cross_anchor_montage_tile_edge": 320,
                "cross_anchor_montage_columns": 3,
                "cross_anchor_montage_crop_padding_ratio": 0.03,
                "cross_anchor_llm2_batch_size": 3,
                "question_truth_min_iou": 0.2,
            }
        ),
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "pages": {
                    "page33": {
                        "regions": [
                            {
                                "truth_id": "T1",
                                "source_bbox_normalized": [0.1, 0.1, 0.3, 0.3],
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config, truth = diagnostic.load_cross_cv_inputs(
        config_path,
        truth_path,
        ["page33"],
    )

    assert config["analysis_max_edge"] == 1600
    assert config["cross_anchor_llm2_localization_runs"] == 2
    assert config["cross_anchor_llm2_retry_min_center_gap_ratio"] == 0.03
    assert config["cross_anchor_llm2_retry_crop_padding_ratio"] == 0.12
    assert (
        config[
            "cross_anchor_llm2_retry_first_pass_min_question_cross_area_ratio"
        ]
        == 1.1
    )
    assert config["cross_anchor_llm2_retry_min_question_cross_area_ratio"] == 1.5
    assert config["cross_anchor_llm2_retry_max_question_cross_iou"] == 0.8
    assert config["cross_anchor_llm2_retry_shared_question_iou_threshold"] == 0.8
    assert (
        config[
            "cross_anchor_llm2_retry_shared_question_min_anchor_distance_ratio"
        ]
        == 0.1
    )
    assert config["cross_anchor_llm2_retry_shared_question_min_group_size"] == 3
    assert config["cross_anchor_llm2_retry_max_requests_per_page"] == 1
    assert truth["page33"][0]["truth_id"] == "T1"
    with pytest.raises(ValueError, match="missing truth page: page34"):
        diagnostic.load_cross_cv_inputs(
            config_path,
            truth_path,
            ["page33", "page34"],
        )
    config["cross_anchor_llm2_localization_runs"] = 0
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="cross_anchor_llm2_localization_runs must be positive",
    ):
        diagnostic.load_cross_cv_inputs(config_path, truth_path, ["page33"])
    config["cross_anchor_llm2_localization_runs"] = 3
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="cross_anchor_llm2_localization_runs must be 1 or 2",
    ):
        diagnostic.load_cross_cv_inputs(config_path, truth_path, ["page33"])


def test_load_cross_cv_inputs_requires_cross_anchor_evaluation_thresholds(tmp_path):
    diagnostic = _load_script_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "analysis_max_edge": 1000,
                "red_min_channel": 60,
                "red_min_excess": 6,
                "axis_line_min_density": 0.65,
                "arm_inner_radius_ratio": 0.006,
                "arm_outer_radius_ratio": 0.03,
                "diagonal_band_ratio": 0.3,
                "arm_min_density": 0.11,
                "center_radius_ratio": 0.004,
                "center_min_density": 0.06,
                "candidate_merge_radius_ratio": 0.04,
                "bbox_padding_ratio": 0.01,
                "truth_match_margin_ratio": 0.01,
                "cross_anchor_question_max_area_ratio": 0.3,
                "cross_anchor_duplicate_question_iou_threshold": 0.2,
                "cross_anchor_retain_uncertain_candidates": True,
                "cross_anchor_high_cv_min_arm_density": 0.25,
                "cross_anchor_high_cv_min_center_density": 0.7,
                "cross_anchor_fallback_merge_iou_threshold": 0.2,
                "cross_anchor_fallback_merge_center_distance_ratio": 0.04,
                "cross_anchor_montage_full_page_max_edge": 1400,
                "cross_anchor_montage_tile_edge": 320,
                "cross_anchor_montage_columns": 3,
                "cross_anchor_montage_crop_padding_ratio": 0.03,
                "question_truth_min_iou": 0.2,
            }
        ),
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "pages": {
                    "page33": {
                        "regions": [
                            {
                                "truth_id": "T1",
                                "source_bbox_normalized": [0.1, 0.1, 0.3, 0.3],
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cross_anchor_question_max_gap_ratio"):
        diagnostic.load_cross_cv_inputs(config_path, truth_path, ["page33"])


def test_cross_candidate_disposition_audit_requires_every_cv_candidate_once():
    diagnostic = _load_script_module()
    result = diagnostic.CrossCandidateVerificationResult.model_validate(
        {
            "verdicts": [
                {"candidate_id": 0, "disposition": "confirmed", "confidence": 0.95},
                {"candidate_id": 1, "disposition": "rejected", "confidence": 0.9},
                {"candidate_id": 1, "disposition": "uncertain", "confidence": 0.5},
            ],
        }
    )

    audit = diagnostic.audit_cross_candidate_dispositions(result, [0, 1, 2])

    assert audit == {
        "valid": False,
        "input_candidate_ids": [0, 1, 2],
        "missing_candidate_ids": [2],
        "unknown_candidate_ids": [],
        "duplicate_candidate_ids": [1],
        "policy": "Every CV candidate must be dispositioned exactly once.",
    }


def test_candidate_verification_rejects_inline_fallback_crosses():
    diagnostic = _load_script_module()

    with pytest.raises(ValueError, match="additional_crosses"):
        diagnostic.CrossCandidateVerificationResult.model_validate(
            {
                "confirmed_crosses": [],
                "rejected_candidate_ids": [],
                "uncertain_candidate_ids": [],
                "additional_crosses": [
                    {"bbox": [0.1, 0.1, 0.2, 0.2], "confidence": 0.9}
                ],
            }
        )


def test_candidate_verification_returns_one_verdict_per_cv_candidate():
    diagnostic = _load_script_module()

    result = diagnostic.CrossCandidateVerificationResult.model_validate(
        {
            "verdicts": [
                {
                    "candidate_id": 0,
                    "disposition": "confirmed",
                    "confidence": 0.95,
                },
                {
                    "candidate_id": 1,
                    "disposition": "uncertain",
                    "confidence": 0.6,
                },
                {
                    "candidate_id": 2,
                    "disposition": "rejected",
                    "confidence": 0.9,
                },
            ]
        }
    )

    assert [item.candidate_id for item in result.verdicts] == [0, 1, 2]
    assert [item.disposition for item in result.verdicts] == [
        "confirmed",
        "uncertain",
        "rejected",
    ]


def test_candidate_verification_runs_are_aggregated_for_recall_with_stability_audit():
    diagnostic = _load_script_module()
    runs = [
        diagnostic.CrossCandidateVerificationResult.model_validate(
            {
                "verdicts": [
                    {"candidate_id": 0, "disposition": "rejected", "confidence": 0.7},
                    {"candidate_id": 1, "disposition": "rejected", "confidence": 0.8},
                    {"candidate_id": 2, "disposition": "rejected", "confidence": 0.9},
                ]
            }
        ),
        diagnostic.CrossCandidateVerificationResult.model_validate(
            {
                "verdicts": [
                    {"candidate_id": 0, "disposition": "confirmed", "confidence": 0.85},
                    {"candidate_id": 1, "disposition": "uncertain", "confidence": 0.55},
                    {"candidate_id": 2, "disposition": "rejected", "confidence": 0.8},
                ]
            }
        ),
        diagnostic.CrossCandidateVerificationResult.model_validate(
            {
                "verdicts": [
                    {"candidate_id": 0, "disposition": "rejected", "confidence": 0.75},
                    {"candidate_id": 1, "disposition": "rejected", "confidence": 0.85},
                    {"candidate_id": 2, "disposition": "rejected", "confidence": 0.88},
                ]
            }
        ),
    ]

    aggregated, audit = diagnostic.aggregate_cross_candidate_verifications(runs)

    assert [item.disposition for item in aggregated.verdicts] == [
        "confirmed",
        "uncertain",
        "rejected",
    ]
    assert [item.confidence for item in aggregated.verdicts] == [0.85, 0.55, 0.9]
    assert audit["run_count"] == 3
    assert audit["unstable_candidate_ids"] == [0, 1]
    assert audit["candidates"][0] == {
        "candidate_id": 0,
        "dispositions": ["rejected", "confirmed", "rejected"],
        "aggregated_disposition": "confirmed",
        "agreement_ratio": 0.666667,
    }


def test_cross_anchors_can_retain_unanimously_rejected_cv_candidates_for_recall():
    diagnostic = _load_script_module()
    verification = diagnostic.CrossCandidateVerificationResult.model_validate(
        {
            "verdicts": [
                {"candidate_id": 0, "disposition": "rejected", "confidence": 0.9},
            ]
        }
    )
    candidates = [
        {
            "candidate_id": 0,
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "min_arm_density": 0.12,
            "center_density": 0.3,
        }
    ]

    anchors = diagnostic.build_cross_anchors(
        verification,
        diagnostic.IndependentCrossScanResult(crosses=[]),
        candidates,
        {
            "cross_anchor_retain_uncertain_candidates": True,
            "cross_anchor_retain_rejected_candidates": True,
            "cross_anchor_high_cv_min_arm_density": 0.25,
            "cross_anchor_high_cv_min_center_density": 0.7,
            "cross_anchor_cv_dedupe_iou_threshold": 0.5,
            "cross_anchor_cv_dedupe_center_distance_ratio": 0.03,
            "cross_anchor_fallback_merge_iou_threshold": 0.2,
            "cross_anchor_fallback_merge_center_distance_ratio": 0.04,
        },
    )

    assert [(item["source"], item["source_candidate_ids"]) for item in anchors] == [
        ("cv_rejected_retained", [0])
    ]


def test_cross_anchors_dedupe_candidate_verdicts_with_local_geometry_only():
    diagnostic = _load_script_module()
    verification = diagnostic.CrossCandidateVerificationResult.model_validate(
        {
            "verdicts": [
                {"candidate_id": 0, "disposition": "confirmed", "confidence": 0.95},
                {"candidate_id": 1, "disposition": "confirmed", "confidence": 0.9},
                {"candidate_id": 2, "disposition": "confirmed", "confidence": 0.85},
            ]
        }
    )
    candidates = [
        {
            "candidate_id": 0,
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "min_arm_density": 0.3,
            "center_density": 0.9,
        },
        {
            "candidate_id": 1,
            "bbox": [0.11, 0.11, 0.21, 0.21],
            "min_arm_density": 0.28,
            "center_density": 0.85,
        },
        {
            "candidate_id": 2,
            "bbox": [0.6, 0.6, 0.7, 0.7],
            "min_arm_density": 0.27,
            "center_density": 0.8,
        },
    ]

    anchors = diagnostic.build_cross_anchors(
        verification,
        diagnostic.IndependentCrossScanResult(crosses=[]),
        candidates,
        {
            "cross_anchor_retain_uncertain_candidates": True,
            "cross_anchor_high_cv_min_arm_density": 0.25,
            "cross_anchor_high_cv_min_center_density": 0.7,
            "cross_anchor_cv_dedupe_iou_threshold": 0.5,
            "cross_anchor_cv_dedupe_center_distance_ratio": 0.03,
            "cross_anchor_fallback_merge_iou_threshold": 0.2,
            "cross_anchor_fallback_merge_center_distance_ratio": 0.04,
        },
    )

    assert [item["source_candidate_ids"] for item in anchors] == [[0, 1], [2]]
    assert anchors[0]["bbox"] == [0.1, 0.1, 0.21, 0.21]
    assert anchors[0]["merge_reason"] == "local_geometry"
    assert anchors[1]["bbox"] == [0.6, 0.6, 0.7, 0.7]
    assert anchors[1]["merge_reason"] is None


def test_cross_anchors_retain_cv_risk_tiers_and_merge_independent_scan():
    diagnostic = _load_script_module()
    verification = diagnostic.CrossCandidateVerificationResult.model_validate(
        {
            "verdicts": [
                {"candidate_id": 0, "disposition": "confirmed", "confidence": 0.95},
                {"candidate_id": 1, "disposition": "rejected", "confidence": 0.9},
                {"candidate_id": 2, "disposition": "rejected", "confidence": 0.9},
                {"candidate_id": 3, "disposition": "uncertain", "confidence": 0.5},
            ],
        }
    )
    independent_scan = diagnostic.IndependentCrossScanResult.model_validate(
        {
            "crosses": [
                {"bbox": [0.11, 0.11, 0.21, 0.21], "confidence": 0.9},
                {"bbox": [0.75, 0.75, 0.85, 0.85], "confidence": 0.85},
            ]
        }
    )
    candidates = [
        {
            "candidate_id": 0,
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "min_arm_density": 0.3,
            "center_density": 0.9,
        },
        {
            "candidate_id": 1,
            "bbox": [0.3, 0.3, 0.4, 0.4],
            "min_arm_density": 0.28,
            "center_density": 0.8,
        },
        {
            "candidate_id": 2,
            "bbox": [0.5, 0.5, 0.6, 0.6],
            "min_arm_density": 0.12,
            "center_density": 0.4,
        },
        {
            "candidate_id": 3,
            "bbox": [0.6, 0.6, 0.7, 0.7],
            "min_arm_density": 0.15,
            "center_density": 0.5,
        },
    ]

    anchors = diagnostic.build_cross_anchors(
        verification,
        independent_scan,
        candidates,
        {
            "cross_anchor_retain_uncertain_candidates": True,
            "cross_anchor_high_cv_min_arm_density": 0.25,
            "cross_anchor_high_cv_min_center_density": 0.7,
            "cross_anchor_cv_dedupe_iou_threshold": 0.5,
            "cross_anchor_cv_dedupe_center_distance_ratio": 0.03,
            "cross_anchor_fallback_merge_iou_threshold": 0.2,
            "cross_anchor_fallback_merge_center_distance_ratio": 0.04,
        },
    )

    assert [(item["source"], item["source_candidate_ids"]) for item in anchors] == [
        ("cv_confirmed", [0]),
        ("cv_high_score_retained", [1]),
        ("cv_uncertain", [3]),
        ("llm_fallback", []),
    ]
    assert anchors[0]["independent_scan_supported"] is True
    assert anchors[1]["independent_scan_supported"] is False
    assert anchors[3]["bbox"] == [0.75, 0.75, 0.85, 0.85]


def test_cross_anchor_assignment_audit_preserves_missing_and_unknown_ids():
    diagnostic = _load_script_module()
    result = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": False,
                    "question_bbox": None,
                    "unmatched_reason": "not clear",
                    "confidence": 0.4,
                },
                {
                    "cross_id": 9,
                    "matched": False,
                    "question_bbox": None,
                    "unmatched_reason": "not clear",
                    "confidence": 0.4,
                },
            ]
        }
    )

    audit = diagnostic.audit_cross_anchor_assignments(result, [0, 1])

    assert audit["valid"] is False
    assert audit["missing_cross_ids"] == [1]
    assert audit["unknown_cross_ids"] == [9]
    assert audit["duplicate_cross_ids"] == []


def test_cross_anchored_question_schema_only_requires_region_localization():
    diagnostic = _load_script_module()

    result = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.1, 0.1, 0.4, 0.4],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                },
                {
                    "cross_id": 1,
                    "matched": False,
                    "question_bbox": None,
                    "unmatched_reason": "not_a_correction_cross",
                    "confidence": 0.8,
                },
            ]
        }
    )

    assert result.items[0].question_bbox == [0.1, 0.1, 0.4, 0.4]
    assert result.items[1].unmatched_reason == "not_a_correction_cross"


def test_cross_anchored_question_schema_requires_reason_when_unmatched():
    diagnostic = _load_script_module()

    with pytest.raises(ValueError, match="unmatched cross anchor requires reason"):
        diagnostic.CrossAnchoredQuestionResult.model_validate(
            {
                "items": [
                    {
                        "cross_id": 1,
                        "matched": False,
                        "question_bbox": None,
                        "unmatched_reason": None,
                        "confidence": 0.8,
                    }
                ]
            }
        )


def test_region_truth_comparison_reports_multiple_crosses_for_same_truth():
    diagnostic = _load_script_module()
    result = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.1, 0.1, 0.4, 0.4],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                },
                {
                    "cross_id": 1,
                    "matched": True,
                    "question_bbox": [0.12, 0.12, 0.38, 0.38],
                    "unmatched_reason": None,
                    "confidence": 0.85,
                },
            ]
        }
    )

    comparison = diagnostic.compare_anchored_questions_to_truth(
        result,
        [
            {
                "truth_id": "T1",
                "source_bbox_normalized": [0.1, 0.1, 0.4, 0.4],
            }
        ],
        min_iou=0.2,
    )

    assert comparison["matched_truth_count"] == 1
    assert comparison["duplicate_truth_candidates"] == [
        {"truth_id": "T1", "cross_ids": [0, 1]}
    ]
    assert comparison["one_to_one_assignments"] == [
        {"cross_id": 0, "truth_id": "T1", "iou": 1.0}
    ]
    assert comparison["unassigned_matched_cross_ids"] == [1]


def test_question_events_preserve_distinct_cross_ids_despite_bbox_overlap():
    diagnostic = _load_script_module()
    result = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.1, 0.1, 0.4, 0.4],
                    "unmatched_reason": None,
                    "confidence": 0.95,
                },
                {
                    "cross_id": 1,
                    "matched": True,
                    "question_bbox": [0.2, 0.1, 0.5, 0.4],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                },
                {
                    "cross_id": 2,
                    "matched": True,
                    "question_bbox": [0.3, 0.1, 0.6, 0.4],
                    "unmatched_reason": None,
                    "confidence": 0.85,
                },
                {
                    "cross_id": 3,
                    "matched": True,
                    "question_bbox": [0.7, 0.7, 0.9, 0.9],
                    "unmatched_reason": None,
                    "confidence": 0.8,
                },
                {
                    "cross_id": 4,
                    "matched": False,
                    "question_bbox": None,
                    "unmatched_reason": "not a correction mark",
                    "confidence": 0.7,
                },
            ]
        }
    )

    audit = diagnostic.cluster_anchored_question_events(result, min_iou=0.2)

    assert audit["event_count"] == 4
    assert audit["unmatched_cross_ids"] == [4]
    assert audit["events"][0] == {
        "event_id": 0,
        "cross_ids": [0],
        "representative_cross_id": 0,
        "question_bboxes": [[0.1, 0.1, 0.4, 0.4]],
        "confidence": 0.95,
    }
    assert audit["events"][1]["cross_ids"] == [1]
    assert audit["events"][2]["cross_ids"] == [2]
    assert audit["events"][3]["cross_ids"] == [3]
    assert [item.cross_id for item in result.items] == [0, 1, 2, 3, 4]


def test_question_event_truth_comparison_uses_best_member_bbox():
    diagnostic = _load_script_module()
    event_audit = {
        "events": [
            {
                "event_id": 0,
                "cross_ids": [0, 1],
                "representative_cross_id": 0,
                "question_bboxes": [
                    [0.0, 0.0, 0.2, 0.2],
                    [0.1, 0.1, 0.4, 0.4],
                ],
                "confidence": 0.9,
            },
            {
                "event_id": 1,
                "cross_ids": [2],
                "representative_cross_id": 2,
                "question_bboxes": [[0.8, 0.8, 0.9, 0.9]],
                "confidence": 0.8,
            },
        ]
    }

    comparison = diagnostic.compare_question_events_to_truth(
        event_audit,
        [
            {"truth_id": "T1", "source_bbox_normalized": [0.1, 0.1, 0.4, 0.4]},
            {"truth_id": "T2", "source_bbox_normalized": [0.5, 0.5, 0.7, 0.7]},
        ],
        min_iou=0.2,
    )

    assert comparison["matched_truth_ids"] == ["T1"]
    assert comparison["missed_truth_ids"] == ["T2"]
    assert comparison["truth_recall"] == 0.5
    assert comparison["minimum_matched_truth_coverage"] == 1.0
    assert comparison["false_event_ids"] == [1]
    assert comparison["assignments"] == [
        {
            "event_id": 0,
            "truth_id": "T1",
            "best_iou": 1.0,
            "truth_coverage": 1.0,
        },
        {
            "event_id": 1,
            "truth_id": None,
            "best_iou": 0.0,
            "truth_coverage": 0.0,
        },
    ]


def test_question_event_truth_comparison_counts_duplicate_truth_events_as_false():
    diagnostic = _load_script_module()
    event_audit = {
        "events": [
            {
                "event_id": 0,
                "question_bboxes": [[0.1, 0.1, 0.4, 0.4]],
            },
            {
                "event_id": 1,
                "question_bboxes": [[0.12, 0.12, 0.38, 0.38]],
            },
        ]
    }

    comparison = diagnostic.compare_question_events_to_truth(
        event_audit,
        [{"truth_id": "T1", "source_bbox_normalized": [0.1, 0.1, 0.4, 0.4]}],
        min_iou=0.2,
    )

    assert comparison["matched_truth_count"] == 1
    assert comparison["truth_recall"] == 1.0
    assert comparison["duplicate_truth_event_ids"] == [1]
    assert comparison["false_event_ids"] == [1]


def test_first_pass_llm2_risk_audit_identifies_true_anchor_localization_failures():
    diagnostic = _load_script_module()
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.05, 0.05, 0.25, 0.25],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                },
                {
                    "cross_id": 1,
                    "matched": True,
                    "question_bbox": [0.35, 0.35, 0.55, 0.55],
                    "unmatched_reason": None,
                    "confidence": 0.8,
                },
                {
                    "cross_id": 2,
                    "matched": False,
                    "question_bbox": None,
                    "unmatched_reason": "not found",
                    "confidence": 0.7,
                },
            ]
        }
    )
    anchors = [
        {"cross_id": 0, "source": "cv_confirmed", "bbox": [0.1, 0.1, 0.2, 0.2]},
        {"cross_id": 1, "source": "cv_confirmed", "bbox": [0.4, 0.4, 0.5, 0.5]},
        {"cross_id": 2, "source": "cv_confirmed", "bbox": [0.7, 0.7, 0.8, 0.8]},
    ]

    audit = diagnostic.audit_first_pass_llm2_localization_risk(
        first_pass,
        anchors,
        {
            "assignments": [
                {"candidate_id": 0, "truth_id": "T1"},
                {"candidate_id": 1, "truth_id": None},
                {"candidate_id": 2, "truth_id": "T3"},
            ]
        },
        {
            "items": [
                {"cross_id": 0, "truth_id": "T2", "meets_threshold": True},
                {"cross_id": 1, "truth_id": None, "meets_threshold": False},
            ]
        },
        {
            "violations": [
                {
                    "cross_id": 0,
                    "reasons": ["question_bbox_not_near_cross"],
                }
            ]
        },
    )

    assert audit["true_anchor_count"] == 2
    assert audit["true_anchor_localization_failure_count"] == 2
    assert audit["false_anchor_matched_count"] == 1
    assert [item["localization_status"] for item in audit["items"]] == [
        "wrong_truth",
        "matched_false_anchor",
        "unmatched_true_anchor",
    ]
    assert audit["items"][0]["question_cross_area_ratio"] == 4.0
    assert audit["items"][0]["question_cross_width_ratio"] == 2.0
    assert audit["items"][0]["question_cross_height_ratio"] == 2.0
    assert audit["items"][0]["cross_center_gap_ratio"] == 0.0
    assert audit["items"][0]["geometry_reasons"] == [
        "question_bbox_not_near_cross"
    ]


def test_question_event_clustering_unions_two_llm2_runs_without_losing_disagreement():
    diagnostic = _load_script_module()
    runs = [
        diagnostic.CrossAnchoredQuestionResult.model_validate(
            {
                "items": [
                    {
                        "cross_id": 0,
                        "matched": True,
                        "question_bbox": [0.1, 0.1, 0.3, 0.3],
                        "unmatched_reason": None,
                        "confidence": 0.9,
                    },
                    {
                        "cross_id": 1,
                        "matched": True,
                        "question_bbox": [0.7, 0.1, 0.9, 0.3],
                        "unmatched_reason": None,
                        "confidence": 0.8,
                    },
                ]
            }
        ),
        diagnostic.CrossAnchoredQuestionResult.model_validate(
            {
                "items": [
                    {
                        "cross_id": 0,
                        "matched": True,
                        "question_bbox": [0.11, 0.11, 0.31, 0.31],
                        "unmatched_reason": None,
                        "confidence": 0.95,
                    },
                    {
                        "cross_id": 1,
                        "matched": True,
                        "question_bbox": [0.4, 0.4, 0.6, 0.6],
                        "unmatched_reason": None,
                        "confidence": 0.9,
                    },
                ]
            }
        ),
    ]

    events = diagnostic.cluster_anchored_question_runs(runs, min_iou=0.2)
    first_events = diagnostic.cluster_anchored_question_events(runs[0], min_iou=0.2)
    truths = [
        {"truth_id": "T1", "source_bbox_normalized": [0.1, 0.1, 0.3, 0.3]},
        {"truth_id": "T2", "source_bbox_normalized": [0.4, 0.4, 0.6, 0.6]},
    ]
    first_truth = diagnostic.compare_question_events_to_truth(
        first_events, truths, min_iou=0.2
    )
    union_truth = diagnostic.compare_question_events_to_truth(
        events, truths, min_iou=0.2
    )
    benefit = diagnostic.compare_llm2_pass_benefit(first_truth, union_truth)

    assert events["run_count"] == 2
    assert events["observation_count"] == 4
    assert events["event_count"] == 2
    assert events["events"][0]["observation_ids"] == ["run-001-cross-0", "run-002-cross-0"]
    assert events["events"][1]["observation_ids"] == ["run-001-cross-1", "run-002-cross-1"]
    assert benefit == {
        "first_pass_matched_truth_count": 1,
        "union_matched_truth_count": 2,
        "union_truth_recall": 1.0,
        "recovered_truth_ids": ["T2"],
        "recovered_truth_count": 1,
        "remaining_missed_truth_ids": [],
        "first_pass_false_event_count": 1,
        "union_false_event_count": 0,
        "additional_false_event_count": 0,
        "net_false_event_delta": -1,
        "first_pass_minimum_matched_truth_coverage": 1.0,
        "union_minimum_matched_truth_coverage": 1.0,
        "truth_recall_delta": 0.5,
    }


def test_question_event_clustering_does_not_bridge_distinct_first_pass_events():
    diagnostic = _load_script_module()
    runs = [
        diagnostic.CrossAnchoredQuestionResult.model_validate(
            {
                "items": [
                    {
                        "cross_id": 0,
                        "matched": True,
                        "question_bbox": [0.0, 0.0, 0.4, 0.4],
                        "unmatched_reason": None,
                        "confidence": 0.9,
                    },
                    {
                        "cross_id": 1,
                        "matched": True,
                        "question_bbox": [0.4, 0.0, 0.8, 0.4],
                        "unmatched_reason": None,
                        "confidence": 0.9,
                    },
                ]
            }
        ),
        diagnostic.CrossAnchoredQuestionResult.model_validate(
            {
                "items": [
                    {
                        "cross_id": 0,
                        "matched": True,
                        "question_bbox": [0.2, 0.0, 0.6, 0.4],
                        "unmatched_reason": None,
                        "confidence": 0.8,
                    },
                    {
                        "cross_id": 1,
                        "matched": True,
                        "question_bbox": [0.4, 0.0, 0.8, 0.4],
                        "unmatched_reason": None,
                        "confidence": 0.8,
                    },
                ]
            }
        ),
    ]

    events = diagnostic.cluster_anchored_question_runs(runs, min_iou=0.2)

    assert events["event_count"] == 2
    assert events["events"][0]["cross_ids"] == [0]
    assert events["events"][1]["cross_ids"] == [1]


def test_llm2_pass_benefit_keeps_first_pass_truth_hits_monotonic():
    diagnostic = _load_script_module()
    first_pass = {
        "matched_truth_count": 2,
        "matched_truth_ids": ["T1", "T2"],
        "missed_truth_ids": [],
        "false_event_ids": [2],
        "minimum_matched_truth_coverage": 0.4,
        "truth_recall": 1.0,
    }
    clustered_union = {
        "matched_truth_count": 1,
        "matched_truth_ids": ["T1"],
        "missed_truth_ids": ["T2"],
        "false_event_ids": [],
        "minimum_matched_truth_coverage": 0.5,
        "truth_recall": 0.5,
    }

    benefit = diagnostic.compare_llm2_pass_benefit(first_pass, clustered_union)

    assert benefit["union_matched_truth_count"] == 2
    assert benefit["remaining_missed_truth_ids"] == []
    assert benefit["truth_recall_delta"] == 0.0
    assert benefit["additional_false_event_count"] == 0
    assert benefit["net_false_event_delta"] == -1


def test_llm2_retry_selection_targets_only_actionable_first_pass_results():
    diagnostic = _load_script_module()
    anchors = [
        {
            "cross_id": 0,
            "source": "cv_rejected_retained",
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "confidence": 0.8,
        },
        {
            "cross_id": 1,
            "source": "cv_rejected_retained",
            "bbox": [0.4, 0.1, 0.5, 0.2],
            "confidence": 0.8,
            "independent_scan_supported": True,
        },
        {
            "cross_id": 2,
            "source": "cv_rejected_retained",
            "bbox": [0.7, 0.1, 0.8, 0.2],
            "confidence": 0.8,
        },
        {
            "cross_id": 3,
            "source": "cv_high_score_retained",
            "bbox": [0.7, 0.7, 0.8, 0.8],
            "confidence": 0.8,
        },
    ]
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.05, 0.05, 0.25, 0.25],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                },
                {
                    "cross_id": 1,
                    "matched": True,
                    "question_bbox": [0.55, 0.05, 0.75, 0.25],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                },
                {
                    "cross_id": 2,
                    "matched": False,
                    "question_bbox": None,
                    "unmatched_reason": "weak CV candidate",
                    "confidence": 0.9,
                },
                {
                    "cross_id": 3,
                    "matched": False,
                    "question_bbox": None,
                    "unmatched_reason": "high CV candidate",
                    "confidence": 0.7,
                },
            ]
        }
    )
    geometry_audit = {
        "violations": [
            {
                "cross_id": 1,
                "reasons": ["question_bbox_not_near_cross"],
                "question_area_ratio": 0.04,
            }
        ]
    }

    selection = diagnostic.select_llm2_retry_anchors(
        first_pass,
        anchors,
        geometry_audit,
        min_center_gap_ratio=0.03,
        min_first_pass_question_cross_area_ratio=1.1,
        max_question_cross_iou=0.8,
        shared_question_iou_threshold=0.8,
        shared_question_min_anchor_distance_ratio=0.1,
        shared_question_min_group_size=3,
        max_retry_requests=1,
    )

    assert [anchor["cross_id"] for anchor in selection["anchors"]] == [3]
    assert selection["trigger_count"] == 1
    assert selection["triggers"] == [
        {
            "cross_id": 3,
            "reasons": ["unmatched_retained_anchor"],
            "cross_center_gap_ratio": None,
        },
    ]
    assert selection["suppressed"] == [
        {
            "cross_id": 1,
            "reasons": ["center_gap_without_independent_anomaly"],
            "candidate_reasons": [
                "cross_center_outside_question_bbox",
                "question_bbox_not_near_cross",
            ],
        },
        {
            "cross_id": 2,
            "reasons": ["retry_budget_exhausted"],
            "candidate_reasons": ["unmatched_retained_anchor"],
        },
    ]


def test_llm2_retry_selection_suppresses_rejected_anchor_without_scan_support():
    diagnostic = _load_script_module()
    anchors = [
        {
            "cross_id": 9,
            "source": "cv_rejected_retained",
            "bbox": [0.25, 0.7, 0.36, 0.78],
            "confidence": 0.75,
            "independent_scan_supported": False,
        }
    ]
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 9,
                    "matched": True,
                    "question_bbox": [0.08, 0.8, 0.32, 0.95],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                }
            ]
        }
    )

    selection = diagnostic.select_llm2_retry_anchors(
        first_pass,
        anchors,
        {
            "violations": [
                {
                    "cross_id": 9,
                    "reasons": ["question_bbox_not_near_cross"],
                    "question_area_ratio": 0.036,
                }
            ]
        },
        min_center_gap_ratio=0.03,
        min_first_pass_question_cross_area_ratio=1.1,
        max_question_cross_iou=0.8,
        shared_question_iou_threshold=0.8,
        shared_question_min_anchor_distance_ratio=0.1,
        shared_question_min_group_size=3,
        max_retry_requests=1,
    )

    assert selection["trigger_count"] == 0
    assert selection["anchors"] == []
    assert selection["suppressed"] == [
        {
            "cross_id": 9,
            "reasons": ["center_gap_without_independent_anomaly"],
            "candidate_reasons": [
                "cross_center_outside_question_bbox",
                "question_bbox_not_near_cross",
            ],
        }
    ]


def test_llm2_retry_selection_bypasses_rejected_suppression_for_cross_copy():
    diagnostic = _load_script_module()
    anchors = [
        {
            "cross_id": 3,
            "source": "cv_rejected_retained",
            "bbox": [0.62, 0.44, 0.73, 0.53],
            "confidence": 0.8,
            "independent_scan_supported": False,
        }
    ]
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 3,
                    "matched": True,
                    "question_bbox": [0.62, 0.44, 0.73, 0.53],
                    "unmatched_reason": None,
                    "confidence": 0.75,
                }
            ]
        }
    )

    selection = diagnostic.select_llm2_retry_anchors(
        first_pass,
        anchors,
        {"violations": []},
        min_center_gap_ratio=0.03,
        min_first_pass_question_cross_area_ratio=1.1,
        max_question_cross_iou=0.8,
        shared_question_iou_threshold=0.8,
        shared_question_min_anchor_distance_ratio=0.1,
        shared_question_min_group_size=3,
        max_retry_requests=1,
    )

    assert [anchor["cross_id"] for anchor in selection["anchors"]] == [3]
    assert selection["suppressed"] == []
    assert selection["triggers"] == [
        {
            "cross_id": 3,
            "reasons": [
                "question_bbox_copies_cross",
                "question_bbox_not_larger_than_cross",
            ],
            "cross_center_gap_ratio": 0.0,
        }
    ]


def test_llm2_retry_selection_does_not_retry_modestly_expanded_question_bbox():
    diagnostic = _load_script_module()
    anchors = [
        {
            "cross_id": 0,
            "source": "cv_rejected_retained",
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "confidence": 0.8,
            "independent_scan_supported": False,
        }
    ]
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.09, 0.09, 0.21, 0.21],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                }
            ]
        }
    )

    selection = diagnostic.select_llm2_retry_anchors(
        first_pass,
        anchors,
        {"violations": []},
        min_center_gap_ratio=0.03,
        min_first_pass_question_cross_area_ratio=1.1,
        max_question_cross_iou=0.8,
        shared_question_iou_threshold=0.8,
        shared_question_min_anchor_distance_ratio=0.1,
        shared_question_min_group_size=3,
        max_retry_requests=1,
    )

    assert selection["trigger_count"] == 0
    assert selection["anchors"] == []


def test_llm2_retry_selection_targets_outlier_anchor_sharing_question_bbox():
    diagnostic = _load_script_module()
    anchors = [
        {
            "cross_id": 3,
            "source": "cv_rejected_retained",
            "bbox": [0.627297, 0.448, 0.733596, 0.529],
            "confidence": 0.8,
            "independent_scan_supported": False,
        },
        {
            "cross_id": 4,
            "source": "cv_rejected_retained",
            "bbox": [0.76378, 0.541, 0.870079, 0.622],
            "confidence": 0.8,
            "independent_scan_supported": False,
        },
        {
            "cross_id": 5,
            "source": "cv_rejected_retained",
            "bbox": [0.821522, 0.544, 0.927822, 0.625],
            "confidence": 0.8,
            "independent_scan_supported": False,
        },
    ]
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": cross_id,
                    "matched": True,
                    "question_bbox": [0.703, 0.443, 0.953, 0.633],
                    "unmatched_reason": None,
                    "confidence": 0.82,
                }
                for cross_id in (3, 4, 5)
            ]
        }
    )

    selection = diagnostic.select_llm2_retry_anchors(
        first_pass,
        anchors,
        {"violations": []},
        min_center_gap_ratio=0.03,
        min_first_pass_question_cross_area_ratio=1.1,
        max_question_cross_iou=0.8,
        shared_question_iou_threshold=0.8,
        shared_question_min_anchor_distance_ratio=0.1,
        shared_question_min_group_size=3,
        max_retry_requests=1,
    )

    assert [anchor["cross_id"] for anchor in selection["anchors"]] == [3]
    assert selection["suppressed"] == []
    assert selection["triggers"][0]["reasons"] == [
        "shared_question_bbox_for_spatially_separated_anchor"
    ]


def test_llm2_retry_selection_does_not_flag_a_two_anchor_shared_question():
    diagnostic = _load_script_module()
    anchors = [
        {
            "cross_id": 0,
            "source": "cv_rejected_retained",
            "bbox": [0.10, 0.10, 0.16, 0.16],
            "confidence": 0.8,
            "independent_scan_supported": False,
        },
        {
            "cross_id": 1,
            "source": "cv_rejected_retained",
            "bbox": [0.24, 0.10, 0.30, 0.16],
            "confidence": 0.8,
            "independent_scan_supported": False,
        },
    ]
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": cross_id,
                    "matched": True,
                    "question_bbox": [0.05, 0.05, 0.35, 0.25],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                }
                for cross_id in (0, 1)
            ]
        }
    )

    selection = diagnostic.select_llm2_retry_anchors(
        first_pass,
        anchors,
        {"violations": []},
        min_center_gap_ratio=0.03,
        min_first_pass_question_cross_area_ratio=1.1,
        max_question_cross_iou=0.8,
        shared_question_iou_threshold=0.8,
        shared_question_min_anchor_distance_ratio=0.1,
        shared_question_min_group_size=3,
        max_retry_requests=1,
    )

    assert selection["anchors"] == []
    assert selection["triggers"] == []


def test_llm2_retry_selection_caps_requests_and_prioritizes_unmatched_anchor():
    diagnostic = _load_script_module()
    anchors = [
        {
            "cross_id": 0,
            "source": "cv_confirmed",
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "confidence": 0.95,
        },
        {
            "cross_id": 1,
            "source": "cv_rejected_retained",
            "bbox": [0.6, 0.6, 0.7, 0.7],
            "confidence": 0.7,
            "independent_scan_supported": False,
        },
    ]
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.1, 0.1, 0.2, 0.2],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                },
                {
                    "cross_id": 1,
                    "matched": False,
                    "question_bbox": None,
                    "unmatched_reason": "未确认红叉",
                    "confidence": 0.6,
                },
            ]
        }
    )

    selection = diagnostic.select_llm2_retry_anchors(
        first_pass,
        anchors,
        {"violations": []},
        min_center_gap_ratio=0.03,
        min_first_pass_question_cross_area_ratio=1.1,
        max_question_cross_iou=0.8,
        shared_question_iou_threshold=0.8,
        shared_question_min_anchor_distance_ratio=0.1,
        shared_question_min_group_size=3,
        max_retry_requests=1,
    )

    assert [anchor["cross_id"] for anchor in selection["anchors"]] == [1]
    assert selection["triggers"][0]["reasons"] == ["unmatched_retained_anchor"]
    assert selection["suppressed"] == [
        {
            "cross_id": 0,
            "reasons": ["retry_budget_exhausted"],
            "candidate_reasons": [
                "question_bbox_copies_cross",
                "question_bbox_not_larger_than_cross",
            ],
        }
    ]


def test_retry_decision_rejects_cross_copy_and_first_pass_duplicate():
    diagnostic = _load_script_module()
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.1, 0.1, 0.3, 0.3],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                },
                {
                    "cross_id": 1,
                    "matched": False,
                    "question_bbox": None,
                    "unmatched_reason": "not found",
                    "confidence": 0.9,
                },
                {
                    "cross_id": 2,
                    "matched": True,
                    "question_bbox": [0.55, 0.55, 0.75, 0.75],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                },
            ]
        }
    )
    retry = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.102, 0.102, 0.198, 0.198],
                    "unmatched_reason": None,
                    "confidence": 0.92,
                },
                {
                    "cross_id": 1,
                    "matched": True,
                    "question_bbox": [0.56, 0.56, 0.76, 0.76],
                    "unmatched_reason": None,
                    "confidence": 0.85,
                },
            ]
        }
    )
    anchors = [
        {"cross_id": 0, "bbox": [0.1, 0.1, 0.2, 0.2]},
        {"cross_id": 1, "bbox": [0.6, 0.6, 0.7, 0.7]},
    ]

    accepted, audit = diagnostic.decide_llm2_retry_results(
        first_pass,
        retry,
        anchors,
        min_question_cross_area_ratio=1.5,
        max_question_cross_iou=0.8,
        duplicate_question_iou_threshold=0.2,
    )

    assert accepted.items == []
    assert audit["accepted_count"] == 0
    assert audit["rejected_count"] == 2
    assert audit["decisions"][0]["reasons"] == [
        "question_bbox_copies_cross",
        "question_bbox_not_larger_than_cross",
    ]
    assert audit["decisions"][1]["reasons"] == [
        "duplicates_first_pass_question"
    ]


def test_retry_decision_accepts_expanded_novel_question_region():
    diagnostic = _load_script_module()
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": False,
                    "question_bbox": None,
                    "unmatched_reason": "not found",
                    "confidence": 0.8,
                }
            ]
        }
    )
    retry = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.35, 0.35, 0.65, 0.65],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                }
            ]
        }
    )

    accepted, audit = diagnostic.decide_llm2_retry_results(
        first_pass,
        retry,
        [{"cross_id": 0, "bbox": [0.45, 0.45, 0.55, 0.55]}],
        min_question_cross_area_ratio=1.5,
        max_question_cross_iou=0.8,
        duplicate_question_iou_threshold=0.2,
    )

    assert [item.cross_id for item in accepted.items] == [0]
    assert audit["accepted_count"] == 1
    assert audit["rejected_count"] == 0
    assert audit["decisions"][0]["accepted"] is True


def test_retry_decision_allows_improved_bbox_for_same_cross():
    diagnostic = _load_script_module()
    first_pass = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.4, 0.4, 0.6, 0.6],
                    "unmatched_reason": None,
                    "confidence": 0.8,
                }
            ]
        }
    )
    retry = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.35, 0.35, 0.65, 0.65],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                }
            ]
        }
    )

    accepted, audit = diagnostic.decide_llm2_retry_results(
        first_pass,
        retry,
        [{"cross_id": 0, "bbox": [0.45, 0.45, 0.55, 0.55]}],
        min_question_cross_area_ratio=1.5,
        max_question_cross_iou=0.8,
        duplicate_question_iou_threshold=0.2,
    )

    assert [item.cross_id for item in accepted.items] == [0]
    assert audit["decisions"][0]["accepted"] is True


def test_retry_crop_maps_anchor_and_question_bbox_between_coordinate_spaces(tmp_path):
    diagnostic = _load_script_module()
    source = tmp_path / "page.jpg"
    output = tmp_path / "retry.jpg"
    Image.new("RGB", (1000, 1000), "white").save(source)
    anchor = {
        "cross_id": 4,
        "source": "cv_rejected_retained",
        "bbox": [0.4, 0.4, 0.5, 0.5],
        "confidence": 0.8,
    }

    crop = diagnostic.write_cross_anchor_retry_crop(
        source,
        output,
        anchor,
        padding_ratio=0.1,
    )
    mapped = diagnostic.map_retry_result_to_source(
        diagnostic.CrossAnchoredQuestionResult.model_validate(
            {
                "items": [
                    {
                        "cross_id": 4,
                        "matched": True,
                        "question_bbox": [0.0, 0.0, 1.0, 1.0],
                        "unmatched_reason": None,
                        "confidence": 0.9,
                    }
                ]
            }
        ),
        crop["source_crop_bbox"],
    )

    assert output.is_file()
    assert crop["source_crop_bbox"] == pytest.approx([0.3, 0.3, 0.6, 0.6])
    assert crop["anchor"]["bbox"] == pytest.approx(
        [1 / 3, 1 / 3, 2 / 3, 2 / 3]
    )
    assert mapped.items[0].question_bbox == pytest.approx([0.3, 0.3, 0.6, 0.6])
    with Image.open(output) as retry_image:
        anchor_corner = retry_image.convert("RGB").getpixel((100, 100))
    assert max(anchor_corner) - min(anchor_corner) <= 5


def test_anchored_question_geometry_requires_bbox_to_contain_cross_center():
    diagnostic = _load_script_module()
    anchors = [
        {
            "cross_id": 0,
            "source": "cv_confirmed",
            "source_candidate_ids": [0],
            "bbox": [0.7, 0.7, 0.8, 0.8],
            "confidence": 0.9,
        }
    ]
    result = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.1, 0.1, 0.4, 0.4],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                }
            ]
        }
    )

    audit = diagnostic.audit_anchored_question_geometry(
        result,
        anchors,
        max_area_ratio=0.3,
        max_gap_ratio=0.03,
    )

    assert audit["valid"] is False
    assert audit["violations"] == [
        {
            "cross_id": 0,
            "reasons": ["question_bbox_not_near_cross"],
            "question_area_ratio": 0.09,
        }
    ]


def test_anchored_question_geometry_allows_adjacent_cross_outside_question_bbox():
    diagnostic = _load_script_module()
    anchors = [
        {
            "cross_id": 0,
            "source": "cv_confirmed",
            "source_candidate_ids": [0],
            "bbox": [0.42, 0.2, 0.48, 0.3],
            "confidence": 0.9,
        }
    ]
    result = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.1, 0.1, 0.4, 0.4],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                }
            ]
        }
    )

    audit = diagnostic.audit_anchored_question_geometry(
        result,
        anchors,
        max_area_ratio=0.3,
        max_gap_ratio=0.03,
    )

    assert audit["valid"] is True
    assert audit["violations"] == []


def test_anchored_question_truth_comparison_uses_question_bbox_iou():
    diagnostic = _load_script_module()
    result = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.1, 0.1, 0.4, 0.4],
                    "unmatched_reason": None,
                    "confidence": 0.9,
                }
            ]
        }
    )
    truths = [
        {"truth_id": "T1", "source_bbox_normalized": [0.1, 0.1, 0.4, 0.4]},
        {"truth_id": "T2", "source_bbox_normalized": [0.6, 0.6, 0.8, 0.8]},
    ]

    comparison = diagnostic.compare_anchored_questions_to_truth(
        result,
        truths,
        min_iou=0.2,
    )

    assert comparison["matched_truth_ids"] == ["T1"]
    assert comparison["missed_truth_ids"] == ["T2"]
    assert comparison["truth_recall"] == 0.5
    assert comparison["duplicate_truth_candidates"] == []
    assert comparison["items"][0] == {
        "cross_id": 0,
        "truth_id": "T1",
        "best_iou": 1.0,
        "meets_threshold": True,
    }


def test_duplicate_anchored_question_audit_flags_overlapping_question_regions():
    diagnostic = _load_script_module()
    common = {
        "matched": True,
        "unmatched_reason": None,
        "confidence": 0.9,
    }
    result = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {"cross_id": 2, "question_bbox": [0.1, 0.1, 0.4, 0.4], **common},
                {"cross_id": 5, "question_bbox": [0.2, 0.2, 0.5, 0.5], **common},
                {"cross_id": 7, "question_bbox": [0.7, 0.7, 0.9, 0.9], **common},
            ]
        }
    )

    audit = diagnostic.audit_duplicate_anchored_questions(
        result,
        min_iou=0.2,
    )

    assert audit == {
        "min_iou": 0.2,
        "duplicate_candidates": [
            {"cross_ids": [2, 5], "question_bbox_iou": 0.285714}
        ],
        "policy": "Duplicate audit records candidates and never rewrites LLM output.",
    }


def test_localized_question_truth_comparison_uses_same_iou_policy_as_new_flow():
    diagnostic = _load_script_module()

    comparison = diagnostic.compare_localized_questions_to_truth(
        [
            {"mark_id": 4, "bbox": [0.1, 0.1, 0.4, 0.4]},
            {"mark_id": 7, "bbox": [0.8, 0.8, 0.9, 0.9]},
        ],
        [
            {"truth_id": "T1", "source_bbox_normalized": [0.1, 0.1, 0.4, 0.4]},
            {"truth_id": "T2", "source_bbox_normalized": [0.5, 0.5, 0.7, 0.7]},
        ],
        min_iou=0.2,
    )

    assert comparison["matched_truth_ids"] == ["T1"]
    assert comparison["missed_truth_ids"] == ["T2"]
    assert comparison["truth_recall"] == 0.5
    assert comparison["items"] == [
        {
            "mark_id": 4,
            "truth_id": "T1",
            "best_iou": 1.0,
            "meets_threshold": True,
        },
        {
            "mark_id": 7,
            "truth_id": None,
            "best_iou": 0.0,
            "meets_threshold": False,
        },
    ]


def test_cross_anchor_experiment_runs_candidate_verification_and_region_localization(
    tmp_path,
):
    diagnostic = _load_script_module()

    source = tmp_path / "page.jpg"
    Image.new("RGB", (200, 200), "white").save(source)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    candidate_overlay = tmp_path / "candidate-overlay.jpg"
    Image.new("RGB", (200, 200), "white").save(candidate_overlay)
    candidates = [
        {
            "candidate_id": 0,
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "center": [0.15, 0.15],
            "min_arm_density": 0.3,
            "center_density": 0.9,
        },
        {
            "candidate_id": 1,
            "bbox": [0.6, 0.6, 0.7, 0.7],
            "center": [0.65, 0.65],
            "min_arm_density": 0.1,
            "center_density": 0.3,
        },
    ]
    calls = []

    class FakeClient:
        def verify_cross_candidates(self, image_path, received_candidates):
            calls.append(("verify", Path(image_path).name, len(received_candidates)))
            return diagnostic.CrossCandidateVerificationResult.model_validate(
                {
                    "verdicts": [
                        {
                            "candidate_id": 0,
                            "disposition": "confirmed",
                            "confidence": 0.95,
                        },
                        {
                            "candidate_id": 1,
                            "disposition": "rejected",
                            "confidence": 0.9,
                        },
                    ],
                }
            )

        def scan_independent_crosses(self, image_path):
            calls.append(("scan", Path(image_path).name))
            return diagnostic.IndependentCrossScanResult.model_validate(
                {
                    "crosses": [
                        {"bbox": [0.6, 0.6, 0.7, 0.7], "confidence": 0.85}
                    ]
                }
            )

        def verify_fallback_crosses(self, image_path, received_candidates):
            calls.append(
                ("verify_fallback", Path(image_path).name, len(received_candidates))
            )
            return diagnostic.CrossCandidateVerificationResult.model_validate(
                {
                    "verdicts": [
                        {"candidate_id": 0, "disposition": "confirmed", "confidence": 0.9}
                    ]
                }
            )

        def locate_cross_anchored_questions(self, image_path, anchors, subject_hint):
            calls.append(("localize", Path(image_path).name, len(anchors), subject_hint))
            anchor = anchors[0]
            if Path(image_path).name.startswith("llm2-retry-cross-"):
                question_bbox = [0.2, 0.2, 0.7, 0.7]
                matched = True
            elif anchor["cross_id"] == 0:
                question_bbox = [0.05, 0.05, 0.3, 0.3]
                matched = True
            else:
                question_bbox = None
                matched = False
            return diagnostic.CrossAnchoredQuestionResult.model_validate(
                {
                    "items": [
                        {
                            "cross_id": anchor["cross_id"],
                            "matched": matched,
                            "question_bbox": question_bbox,
                            "unmatched_reason": None if matched else "未确认红叉",
                            "confidence": 0.9,
                        },
                    ]
                }
            )

    summary = diagnostic.run_cross_anchor_experiment(
        image_path=source,
        case_dir=case_dir,
        client=FakeClient(),
        cv_candidates=candidates,
        candidate_overlay_path=candidate_overlay,
        truth_regions=[
            {"truth_id": "T1", "source_bbox_normalized": [0.05, 0.05, 0.3, 0.3]},
            {"truth_id": "T2", "source_bbox_normalized": [0.5, 0.5, 0.75, 0.75]},
        ],
        config={
            "cross_anchor_question_max_area_ratio": 0.3,
            "cross_anchor_question_max_gap_ratio": 0.03,
            "cross_anchor_duplicate_question_iou_threshold": 0.2,
            "question_truth_min_iou": 0.2,
            "cross_anchor_llm1_verification_runs": 1,
            "cross_anchor_llm2_localization_runs": 2,
            "cross_anchor_llm2_retry_min_center_gap_ratio": 0.03,
            "cross_anchor_llm2_retry_crop_padding_ratio": 0.2,
            "cross_anchor_llm2_retry_first_pass_min_question_cross_area_ratio": 1.1,
            "cross_anchor_llm2_retry_min_question_cross_area_ratio": 1.5,
            "cross_anchor_llm2_retry_max_question_cross_iou": 0.8,
            "cross_anchor_llm2_retry_shared_question_iou_threshold": 0.8,
            "cross_anchor_llm2_retry_shared_question_min_anchor_distance_ratio": 0.1,
            "cross_anchor_llm2_retry_shared_question_min_group_size": 3,
            "cross_anchor_llm2_retry_max_requests_per_page": 1,
            "cross_anchor_fallback_generates_anchors": False,
            "cross_anchor_retain_uncertain_candidates": True,
            "cross_anchor_retain_rejected_candidates": True,
            "cross_anchor_retain_uncertain_fallback_candidates": True,
            "cross_anchor_high_cv_min_arm_density": 0.25,
            "cross_anchor_high_cv_min_center_density": 0.7,
            "cross_anchor_cv_dedupe_iou_threshold": 0.5,
            "cross_anchor_cv_dedupe_center_distance_ratio": 0.03,
            "cross_anchor_fallback_merge_iou_threshold": 0.2,
            "cross_anchor_fallback_merge_center_distance_ratio": 0.04,
            "cross_anchor_montage_full_page_max_edge": 200,
            "cross_anchor_montage_tile_edge": 120,
            "cross_anchor_montage_columns": 2,
            "cross_anchor_montage_crop_padding_ratio": 0.03,
            "cross_anchor_llm2_batch_size": 1,
        },
        subject_hint="chinese",
    )

    assert [call[0] for call in calls] == [
        "verify",
        "scan",
        "verify_fallback",
        "localize",
        "localize",
        "localize",
    ]
    assert calls[0][1] == "candidate-montage.jpg"
    assert calls[2][1] == "fallback-candidate-montage.jpg"
    assert [call[2] for call in calls if call[0] == "localize"] == [1, 1, 1]
    assert calls[-1][1] == "llm2-retry-cross-001.jpg"
    assert summary["llm1_verification_run_count"] == 1
    assert summary["llm1_unstable_candidate_count"] == 0
    assert summary["llm1_confirmed_cross_count"] == 2
    assert summary["llm1_cv_confirmed_cross_count"] == 1
    assert summary["llm1_cv_uncertain_retained_count"] == 0
    assert summary["llm1_cv_rejected_retained_count"] == 1
    assert summary["llm1_model_confirmed_truth_matched_count"] == 1
    assert summary["llm1_model_confirmed_truth_recall"] == 0.5
    assert summary["llm1_truth_matched_count"] == 2
    assert summary["llm1_truth_recall"] == 1.0
    assert summary["llm1_fallback_cross_count"] == 0
    assert summary["llm1_fallback_verified_count"] == 1
    assert summary["llm1_independent_supported_count"] == 1
    assert summary["llm2_localization_run_count"] == 2
    assert summary["first_pass_stable_question_event_count"] == 1
    assert summary["first_pass_stable_truth_recall"] == 0.5
    assert summary["stable_question_event_count"] == 2
    assert summary["stable_truth_recall"] == 1.0
    assert summary["llm2_second_pass_recovered_truth_count"] == 1
    assert summary["llm2_second_pass_recovered_truth_ids"] == ["T2"]
    assert summary["llm2_second_pass_additional_false_event_count"] == 0
    assert summary["first_pass_minimum_matched_truth_coverage"] == 1.0
    assert summary["stable_minimum_matched_truth_coverage"] == 1.0
    assert summary["llm2_retry_trigger_count"] == 1
    assert summary["llm2_retry_request_count"] == 1
    assert summary["llm2_retry_accepted_count"] == 1
    assert summary["llm2_retry_rejected_count"] == 0
    assert summary["llm2_retry_suppressed_count"] == 0
    assert summary["llm2_first_pass_true_anchor_localization_failure_count"] == 1
    assert summary["llm2_first_pass_false_anchor_matched_count"] == 0
    assert summary["llm_request_count"] == 6
    assert summary["timings_ms"]["total"] >= 0
    assert summary["timings_ms"]["llm1_candidate_verification"] >= 0
    assert summary["timings_ms"]["llm2_localization"] >= 0
    assert summary["timings_ms"]["llm2_localization_run_001"] >= 0
    assert summary["timings_ms"]["llm2_localization_run_002"] >= 0
    experiment_dir = case_dir / "cross-anchor-experiment"
    assert (experiment_dir / "llm1-candidate-verification.json").is_file()
    assert (experiment_dir / "llm1-candidate-verification-run-001.json").is_file()
    assert not (experiment_dir / "llm1-candidate-verification-run-002.json").exists()
    assert (experiment_dir / "llm1-candidate-stability-audit.json").is_file()
    assert (experiment_dir / "candidate-montage.jpg").is_file()
    assert (experiment_dir / "llm1-independent-scan.json").is_file()
    assert (experiment_dir / "fallback-candidate-montage.jpg").is_file()
    assert (experiment_dir / "llm1-fallback-candidate-verification.json").is_file()
    assert (experiment_dir / "llm1-candidate-membership-audit.json").is_file()
    assert (experiment_dir / "llm1-anchor-merge-audit.json").is_file()
    assert (experiment_dir / "confirmed-crosses.json").is_file()
    assert (experiment_dir / "confirmed-crosses-overlay.jpg").is_file()
    assert (experiment_dir / "llm2-run-001-batch-001-overlay.jpg").is_file()
    assert (experiment_dir / "llm2-run-001-batch-002-overlay.jpg").is_file()
    assert (experiment_dir / "llm2-retry-cross-001.jpg").is_file()
    assert (experiment_dir / "llm2-retry-selection.json").is_file()
    assert (experiment_dir / "llm2-retry-decision-audit.json").is_file()
    assert (experiment_dir / "llm2-accepted-retry-questions.json").is_file()
    assert (experiment_dir / "llm1-truth-comparison.json").is_file()
    assert (experiment_dir / "llm1-model-confirmed-truth-comparison.json").is_file()
    assert (experiment_dir / "llm1-truth-multiplicity-audit.json").is_file()
    assert (experiment_dir / "llm2-anchored-questions.json").is_file()
    assert (experiment_dir / "llm2-anchored-questions-run-001.json").is_file()
    assert (experiment_dir / "llm2-anchored-questions-run-002.json").is_file()
    assert (experiment_dir / "llm2-pass-benefit.json").is_file()
    assert (experiment_dir / "llm2-first-pass-risk-audit.json").is_file()
    benefit = json.loads(
        (experiment_dir / "llm2-pass-benefit.json").read_text("utf-8")
    )
    assert benefit["second_pass_elapsed_ms"] >= 0
    assert benefit["total_llm2_elapsed_ms"] >= benefit["second_pass_elapsed_ms"]
    assert (experiment_dir / "llm2-cross-assignment-audit.json").is_file()
    assert (experiment_dir / "question-geometry-audit.json").is_file()
    assert (experiment_dir / "duplicate-question-audit.json").is_file()
    assert (experiment_dir / "truth-comparison.json").is_file()
    assert (experiment_dir / "stable-question-events.json").is_file()
    assert (experiment_dir / "stable-question-events-first-pass.json").is_file()
    assert (
        experiment_dir / "stable-question-events-first-pass-truth-comparison.json"
    ).is_file()
    assert (experiment_dir / "stable-question-events-truth-comparison.json").is_file()
    assert (experiment_dir / "timings.json").is_file()
    assert not (experiment_dir / "ocr-audit.json").exists()
    assert (experiment_dir / "summary.json").is_file()


def test_cross_anchor_experiment_stops_before_llm2_on_invalid_candidate_membership(
    tmp_path,
):
    diagnostic = _load_script_module()
    source = tmp_path / "page.jpg"
    Image.new("RGB", (100, 100), "white").save(source)
    overlay = tmp_path / "overlay.jpg"
    Image.new("RGB", (100, 100), "white").save(overlay)

    class FakeClient:
        def verify_cross_candidates(self, _image_path, _candidates):
            return diagnostic.CrossCandidateVerificationResult.model_validate(
                {"verdicts": []}
            )

        def locate_cross_anchored_questions(self, *_args):
            raise AssertionError("LLM2 must not run after invalid LLM1 membership")

        def scan_independent_crosses(self, *_args):
            raise AssertionError("fallback scan must not run after invalid membership")

    with pytest.raises(ValueError, match="candidate disposition audit failed"):
        diagnostic.run_cross_anchor_experiment(
            image_path=source,
            case_dir=tmp_path / "case",
            client=FakeClient(),
            cv_candidates=[
                {
                    "candidate_id": 0,
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                    "center": [0.15, 0.15],
                    "min_arm_density": 0.1,
                    "center_density": 0.3,
                }
            ],
            candidate_overlay_path=overlay,
            truth_regions=[],
            config={
                "cross_anchor_question_max_area_ratio": 0.3,
                "cross_anchor_question_max_gap_ratio": 0.03,
                "cross_anchor_duplicate_question_iou_threshold": 0.2,
                "question_truth_min_iou": 0.2,
                "cross_anchor_retain_uncertain_candidates": True,
                "cross_anchor_high_cv_min_arm_density": 0.25,
                "cross_anchor_high_cv_min_center_density": 0.7,
                "cross_anchor_cv_dedupe_iou_threshold": 0.5,
                "cross_anchor_cv_dedupe_center_distance_ratio": 0.03,
                "cross_anchor_fallback_merge_iou_threshold": 0.2,
                "cross_anchor_fallback_merge_center_distance_ratio": 0.04,
                "cross_anchor_montage_full_page_max_edge": 200,
                "cross_anchor_montage_tile_edge": 120,
                "cross_anchor_montage_columns": 2,
                "cross_anchor_montage_crop_padding_ratio": 0.03,
            },
            subject_hint="chinese",
        )

    audit_path = (
        tmp_path
        / "case"
        / "cross-anchor-experiment"
        / "llm1-candidate-membership-audit.json"
    )
    assert json.loads(audit_path.read_text("utf-8"))["missing_candidate_ids"] == [0]


def test_cross_anchor_experiment_skips_llm2_when_no_cross_is_confirmed(tmp_path):
    diagnostic = _load_script_module()
    source = tmp_path / "page.jpg"
    Image.new("RGB", (100, 100), "white").save(source)
    overlay = tmp_path / "overlay.jpg"
    Image.new("RGB", (100, 100), "white").save(overlay)

    class FakeClient:
        def verify_cross_candidates(self, _image_path, _candidates):
            return diagnostic.CrossCandidateVerificationResult.model_validate(
                {
                    "verdicts": [
                        {"candidate_id": 0, "disposition": "rejected", "confidence": 0.9}
                    ],
                }
            )

        def locate_cross_anchored_questions(self, *_args):
            raise AssertionError("LLM2 must not run without a confirmed cross")

        def scan_independent_crosses(self, _image_path):
            return diagnostic.IndependentCrossScanResult(crosses=[])

    summary = diagnostic.run_cross_anchor_experiment(
        image_path=source,
        case_dir=tmp_path / "case",
        client=FakeClient(),
        cv_candidates=[
            {
                "candidate_id": 0,
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "center": [0.15, 0.15],
                "min_arm_density": 0.1,
                "center_density": 0.3,
            }
        ],
        candidate_overlay_path=overlay,
        truth_regions=[
            {"truth_id": "T1", "source_bbox_normalized": [0.1, 0.1, 0.3, 0.3]}
        ],
        config={
            "cross_anchor_question_max_area_ratio": 0.3,
            "cross_anchor_question_max_gap_ratio": 0.03,
            "cross_anchor_duplicate_question_iou_threshold": 0.2,
            "question_truth_min_iou": 0.2,
            "cross_anchor_llm2_retry_min_center_gap_ratio": 0.03,
            "cross_anchor_llm2_retry_crop_padding_ratio": 0.12,
            "cross_anchor_llm2_retry_first_pass_min_question_cross_area_ratio": 1.1,
            "cross_anchor_llm2_retry_min_question_cross_area_ratio": 1.5,
            "cross_anchor_llm2_retry_max_question_cross_iou": 0.8,
            "cross_anchor_llm2_retry_shared_question_iou_threshold": 0.8,
            "cross_anchor_llm2_retry_shared_question_min_anchor_distance_ratio": 0.1,
            "cross_anchor_llm2_retry_shared_question_min_group_size": 3,
            "cross_anchor_llm2_retry_max_requests_per_page": 1,
            "cross_anchor_retain_uncertain_candidates": True,
            "cross_anchor_high_cv_min_arm_density": 0.25,
            "cross_anchor_high_cv_min_center_density": 0.7,
            "cross_anchor_cv_dedupe_iou_threshold": 0.5,
            "cross_anchor_cv_dedupe_center_distance_ratio": 0.03,
            "cross_anchor_fallback_merge_iou_threshold": 0.2,
            "cross_anchor_fallback_merge_center_distance_ratio": 0.04,
            "cross_anchor_montage_full_page_max_edge": 200,
            "cross_anchor_montage_tile_edge": 120,
            "cross_anchor_montage_columns": 2,
            "cross_anchor_montage_crop_padding_ratio": 0.03,
            "cross_anchor_llm2_batch_size": 2,
        },
        subject_hint="chinese",
    )

    assert summary["llm1_confirmed_cross_count"] == 0
    assert summary["llm2_matched_question_count"] == 0
    assert summary["truth_recall"] == 0.0
    result_path = (
        tmp_path
        / "case"
        / "cross-anchor-experiment"
        / "llm2-anchored-questions.json"
    )
    assert json.loads(result_path.read_text("utf-8")) == {"items": []}


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


def test_recording_client_independent_mark_detection_returns_recorded_result(tmp_path):
    diagnostic = _load_script_module()
    source = tmp_path / "page.jpg"
    Image.new("RGB", (40, 40), "white").save(source)
    calls = []

    class FakeVisionClient:
        max_edge = 100
        jpeg_quality = 90

        def _request(self, payload, result_model, request_diagnostic):
            calls.append((payload, result_model, request_diagnostic))
            return {"error_marks": []}

    recording_client = diagnostic.RecordingVisionClient(
        FakeVisionClient(),
        diagnostic.ExchangeRecorder(tmp_path / "recording"),
    )

    result = recording_client.detect_independent_complete_marks(str(source))

    assert result == {"error_marks": []}
    assert len(calls) == 1
    call_dir = tmp_path / "recording" / "llm-calls" / "call-001-independent_complete_mark_detection"
    assert json.loads((call_dir / "result.json").read_text("utf-8")) == {
        "error_marks": []
    }


def test_recording_client_independent_cross_scan_returns_recorded_result(tmp_path):
    diagnostic = _load_script_module()
    source = tmp_path / "page.jpg"
    Image.new("RGB", (40, 40), "white").save(source)
    calls = []

    class FakeVisionClient:
        max_edge = 100
        jpeg_quality = 90

        def _request(self, payload, result_model, request_diagnostic):
            calls.append((payload, result_model, request_diagnostic))
            return diagnostic.IndependentCrossScanResult.model_validate(
                {"crosses": [{"bbox": [0.1, 0.1, 0.2, 0.2], "confidence": 0.9}]}
            )

    recording_client = diagnostic.RecordingVisionClient(
        FakeVisionClient(),
        diagnostic.ExchangeRecorder(tmp_path / "recording"),
    )

    result = recording_client.scan_independent_crosses(str(source))

    assert result.crosses[0].bbox == [0.1, 0.1, 0.2, 0.2]
    assert len(calls) == 1
    call_dir = tmp_path / "recording" / "llm-calls" / "call-001-independent_cross_scan"
    assert json.loads((call_dir / "result.json").read_text("utf-8"))["crosses"] == [
        {"bbox": [0.1, 0.1, 0.2, 0.2], "confidence": 0.9}
    ]


def test_recording_client_reverifies_fallback_candidates_independently(tmp_path):
    diagnostic = _load_script_module()
    montage = tmp_path / "fallback-montage.jpg"
    Image.new("RGB", (40, 40), "white").save(montage)
    calls = []

    class FakeVisionClient:
        max_edge = 100
        jpeg_quality = 90

        def _request(self, payload, result_model, request_diagnostic):
            calls.append((payload, result_model, request_diagnostic))
            return diagnostic.CrossCandidateVerificationResult.model_validate(
                {
                    "verdicts": [
                        {
                            "candidate_id": 7,
                            "disposition": "confirmed",
                            "confidence": 0.9,
                        }
                    ]
                }
            )

    recording_client = diagnostic.RecordingVisionClient(
        FakeVisionClient(),
        diagnostic.ExchangeRecorder(tmp_path / "recording"),
    )
    candidates = [{"candidate_id": 7, "bbox": [0.1, 0.1, 0.2, 0.2]}]

    result = recording_client.verify_fallback_crosses(str(montage), candidates)

    assert result.verdicts[0].candidate_id == 7
    assert len(calls) == 1
    assert '"candidate_id": 7' in calls[0][0]["prompt"]
    call_dir = (
        tmp_path
        / "recording"
        / "llm-calls"
        / "call-001-fallback_cross_candidate_verification"
    )
    assert json.loads((call_dir / "result.json").read_text("utf-8")) == {
        "verdicts": [
            {
                "candidate_id": 7,
                "disposition": "confirmed",
                "confidence": 0.9,
            }
        ]
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
    assert "圈叉归属异常" in report
    assert "重复primitive候选" in report
    assert "跨单元圈叉候选" in report
    assert (
        "| page34 | 1 | 12 | 4 | None | None | None | None | None | None | "
        "未运行 |"
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


def test_stable_event_normalizes_empty_component_bbox_and_cross_alias():
    diagnostic = _load_script_module()

    result = diagnostic.StableEventResult.model_validate(
        {
            "events": [
                {
                    "event_id": 0,
                    "primitive_ids": [0],
                    "event_type": "cross_only",
                    "bbox": [0.2, 0.2, 0.4, 0.4],
                    "cross_bbox": [0.2, 0.2, 0.4, 0.4],
                    "circle_bbox": [0, 0, 0, 0],
                    "confidence": 0.9,
                }
            ],
            "unassigned_primitive_ids": [],
        }
    )

    assert result.events[0].event_type == "cross"
    assert result.events[0].circle_bbox is None


def test_stable_event_keeps_multi_primitive_response_for_diagnostic_audit():
    diagnostic = _load_script_module()

    result = diagnostic.StableEventResult.model_validate(
        {
            "events": [
                {
                    "event_id": 0,
                    "primitive_ids": [0, 1, 2],
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

    assert result.events[0].primitive_ids == [0, 1, 2]


def test_primitive_membership_audit_reports_wrong_cross_circle_types():
    diagnostic = _load_script_module()
    primitives = diagnostic.MarkDetectionResult.model_validate(
        {
            "error_marks": [
                {
                    "mark_id": 0,
                    "mark_type": "cross",
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                    "cross_bbox": None,
                    "circle_bbox": None,
                    "confidence": 0.9,
                },
                {
                    "mark_id": 1,
                    "mark_type": "cross",
                    "bbox": [0.2, 0.1, 0.3, 0.2],
                    "cross_bbox": None,
                    "circle_bbox": None,
                    "confidence": 0.9,
                },
            ]
        }
    ).error_marks
    result = diagnostic.StableEventResult.model_validate(
        {
            "events": [
                {
                    "event_id": 0,
                    "primitive_ids": [0, 1],
                    "event_type": "cross_circle",
                    "bbox": [0.1, 0.1, 0.3, 0.2],
                    "cross_bbox": [0.2, 0.1, 0.3, 0.2],
                    "circle_bbox": [0.1, 0.1, 0.2, 0.2],
                    "confidence": 0.9,
                }
            ],
            "unassigned_primitive_ids": [],
        }
    )

    violations = diagnostic.audit_stable_event_primitive_membership(
        result,
        primitives,
    )

    assert violations == [
        {
            "event_id": 0,
            "event_type": "cross_circle",
            "primitive_ids": [0, 1],
            "primitive_types": ["cross", "cross"],
            "expected_primitive_types": ["circle", "cross"],
        }
    ]


def test_independent_mark_prompt_forbids_external_cv_coordinates_and_fragments():
    diagnostic = _load_script_module()

    prompt = diagnostic.INDEPENDENT_COMPLETE_MARK_PROMPT

    assert "不接收也不得推测任何外部候选坐标" in prompt
    assert "不得把圆弧、叉的一条笔画或孤立红线作为独立标记" in prompt
    assert "教师文字批注必须标为 annotation" in prompt
    assert "local_red_regions" not in prompt


def test_cross_anchored_prompt_uses_circle_only_as_localization_evidence():
    diagnostic = _load_script_module()

    prompt = diagnostic.CROSS_ANCHORED_QUESTION_PROMPT

    assert "红叉是决定性主锚点" in prompt
    assert "红圈作为辅助证据" in prompt
    assert "区分相邻作答单元" in prompt
    assert "不得仅凭红圈或教师批注新增错题" in prompt
    assert "无需单独返回红圈或教师批注" in prompt


def test_stable_event_prompt_unassigns_teacher_annotations_and_forbids_distant_pairs():
    diagnostic = _load_script_module()

    prompt = diagnostic.STABLE_EVENT_CONSOLIDATION_PROMPT

    assert "教师文字批注" in prompt
    assert "unassigned_primitive_ids" in prompt
    assert "空间明显分离" in prompt


def test_duplicate_primitive_audit_only_reports_overlapping_same_type_marks():
    diagnostic = _load_script_module()
    primitives = diagnostic.MarkDetectionResult.model_validate(
        {
            "error_marks": [
                {
                    "mark_id": 0,
                    "mark_type": "circle",
                    "bbox": [0.1, 0.1, 0.5, 0.5],
                    "cross_bbox": None,
                    "circle_bbox": None,
                    "confidence": 0.9,
                },
                {
                    "mark_id": 1,
                    "mark_type": "circle",
                    "bbox": [0.2, 0.2, 0.45, 0.45],
                    "cross_bbox": None,
                    "circle_bbox": None,
                    "confidence": 0.8,
                },
                {
                    "mark_id": 2,
                    "mark_type": "cross",
                    "bbox": [0.2, 0.2, 0.45, 0.45],
                    "cross_bbox": None,
                    "circle_bbox": None,
                    "confidence": 0.8,
                },
            ]
        }
    ).error_marks

    candidates = diagnostic.find_duplicate_primitive_candidates(
        primitives,
        containment_threshold=0.8,
    )

    assert candidates == [
        {
            "primitive_ids": [0, 1],
            "mark_type": "circle",
            "intersection_over_smaller_area": 1.0,
        }
    ]


def test_cross_circle_geometry_audit_reports_spatially_distant_pair():
    diagnostic = _load_script_module()
    result = diagnostic.StableEventResult.model_validate(
        {
            "events": [
                {
                    "event_id": 0,
                    "primitive_ids": [4, 7],
                    "event_type": "cross_circle",
                    "bbox": [0.4, 0.2, 0.7, 0.7],
                    "cross_bbox": [0.4, 0.2, 0.5, 0.3],
                    "circle_bbox": [0.5, 0.6, 0.7, 0.7],
                    "confidence": 0.9,
                },
                {
                    "event_id": 1,
                    "primitive_ids": [8],
                    "event_type": "cross",
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                    "cross_bbox": [0.1, 0.1, 0.2, 0.2],
                    "circle_bbox": None,
                    "confidence": 0.9,
                },
            ],
            "unassigned_primitive_ids": [],
        }
    )

    candidates = diagnostic.audit_cross_circle_geometry(
        result,
        max_center_distance=0.2,
    )

    assert candidates == [
        {
            "event_id": 0,
            "primitive_ids": [4, 7],
            "center_distance": 0.4272,
            "max_center_distance": 0.2,
        }
    ]


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
            "duplicate_primitive_candidates": [{"primitive_ids": [0, 1]}],
            "cross_circle_geometry_candidates": [{"event_id": 0}],
            "primitive_membership_violations": [],
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
    assert checkpoints["stable_duplicate_primitive_candidate_count"] == 1
    assert checkpoints["stable_cross_circle_geometry_candidate_count"] == 1
    assert checkpoints["stable_primitive_membership_violation_count"] == 0
    assert checkpoints["stable_duplicate_event_candidate_count"] == 0
    assert checkpoints["stable_uncovered_component_count"] == 1


def test_comparison_summary_keeps_cross_anchor_stage_counts_separate():
    diagnostic = _load_script_module()

    summary = diagnostic.build_summary(
        label="page33",
        expected_count=6,
        cv={"raw_component_count": 16, "evidence_group_count": 7},
        pipeline=None,
        pipeline_truth_comparison={
            "truth_count": 6,
            "matched_truth_count": 4,
            "truth_recall": 0.666667,
        },
        cross_anchor_experiment={
            "cv_candidate_count": 12,
            "llm1_confirmed_cross_count": 7,
            "llm1_cv_confirmed_cross_count": 6,
            "llm1_cv_uncertain_retained_count": 1,
            "llm1_cv_high_score_retained_count": 2,
            "llm1_fallback_cross_count": 1,
            "llm1_fallback_verified_count": 1,
            "llm1_independent_scan_count": 5,
            "llm1_independent_supported_count": 4,
            "llm1_rejected_candidate_count": 5,
            "llm1_uncertain_candidate_count": 0,
            "llm1_candidate_audit_valid": True,
            "llm1_local_geometry_merge_count": 2,
            "llm1_truth_matched_count": 6,
            "llm1_truth_recall": 1.0,
            "llm1_model_confirmed_truth_matched_count": 4,
            "llm1_model_confirmed_truth_recall": 0.666667,
            "llm1_false_cross_count": 1,
            "llm1_duplicate_truth_candidate_count": 2,
            "llm2_localization_run_count": 2,
            "llm2_retry_trigger_count": 2,
            "llm2_retry_request_count": 2,
            "llm2_retry_suppressed_count": 1,
            "llm2_retry_accepted_count": 1,
            "llm2_retry_rejected_count": 1,
            "llm2_first_pass_true_anchor_localization_failure_count": 2,
            "llm2_first_pass_false_anchor_matched_count": 1,
            "llm2_matched_question_count": 6,
            "llm2_unmatched_cross_count": 1,
            "llm2_assignment_audit_valid": True,
            "geometry_violation_count": 1,
            "duplicate_question_candidate_count": 2,
            "duplicate_truth_candidate_count": 1,
            "truth_matched_count": 6,
            "truth_count": 6,
            "truth_recall": 1.0,
            "first_pass_stable_question_event_count": 6,
            "first_pass_stable_truth_matched_count": 5,
            "first_pass_stable_truth_recall": 0.833333,
            "first_pass_stable_false_event_count": 1,
            "stable_question_event_count": 7,
            "stable_truth_matched_count": 6,
            "stable_truth_recall": 1.0,
            "stable_false_event_count": 1,
            "llm2_second_pass_recovered_truth_count": 1,
            "llm2_second_pass_recovered_truth_ids": ["T4"],
            "llm2_second_pass_additional_false_event_count": 0,
            "llm_request_count": 12,
            "content_ocr_status": "not_run",
        },
    )

    checkpoints = summary["checkpoints"]
    assert summary["cross_anchor_experiment_status"] == "completed"
    assert checkpoints["pipeline_truth_matched_count"] == 4
    assert checkpoints["pipeline_truth_recall"] == 0.666667
    assert checkpoints["cross_anchor_cv_candidate_count"] == 12
    assert checkpoints["cross_anchor_confirmed_cross_count"] == 7
    assert checkpoints["cross_anchor_uncertain_retained_count"] == 1
    assert checkpoints["cross_anchor_high_score_retained_count"] == 2
    assert checkpoints["cross_anchor_fallback_cross_count"] == 1
    assert checkpoints["cross_anchor_fallback_verified_count"] == 1
    assert checkpoints["cross_anchor_independent_scan_count"] == 5
    assert checkpoints["cross_anchor_independent_supported_count"] == 4
    assert checkpoints["cross_anchor_local_geometry_merge_count"] == 2
    assert checkpoints["cross_anchor_llm1_truth_recall"] == 1.0
    assert checkpoints["cross_anchor_llm1_model_confirmed_truth_recall"] == 0.666667
    assert checkpoints["cross_anchor_llm1_false_cross_count"] == 1
    assert checkpoints["cross_anchor_llm1_duplicate_truth_candidate_count"] == 2
    assert checkpoints["cross_anchor_matched_question_count"] == 6
    assert checkpoints["cross_anchor_llm2_localization_run_count"] == 2
    assert checkpoints["cross_anchor_llm2_retry_trigger_count"] == 2
    assert checkpoints["cross_anchor_llm2_retry_request_count"] == 2
    assert checkpoints["cross_anchor_llm2_retry_suppressed_count"] == 1
    assert checkpoints["cross_anchor_llm2_retry_accepted_count"] == 1
    assert checkpoints["cross_anchor_llm2_retry_rejected_count"] == 1
    assert (
        checkpoints[
            "cross_anchor_llm2_first_pass_true_anchor_localization_failure_count"
        ]
        == 2
    )
    assert checkpoints["cross_anchor_first_pass_stable_truth_recall"] == 0.833333
    assert checkpoints["cross_anchor_stable_truth_recall"] == 1.0
    assert checkpoints["cross_anchor_second_pass_recovered_truth_count"] == 1
    assert checkpoints["cross_anchor_second_pass_recovered_truth_ids"] == ["T4"]
    assert checkpoints["cross_anchor_second_pass_additional_false_event_count"] == 0
    assert checkpoints["cross_anchor_geometry_violation_count"] == 1
    assert checkpoints["cross_anchor_duplicate_question_candidate_count"] == 2
    assert checkpoints["cross_anchor_duplicate_truth_candidate_count"] == 1
    assert checkpoints["cross_anchor_truth_recall"] == 1.0
    assert checkpoints["cross_anchor_content_ocr_status"] == "not_run"


def test_report_includes_cross_anchor_comparison_columns(tmp_path):
    diagnostic = _load_script_module()
    summary = diagnostic.build_summary(
        label="page33",
        expected_count=6,
        cv={"raw_component_count": 16, "evidence_group_count": 7},
        pipeline=None,
        cross_anchor_experiment={
            "cv_candidate_count": 12,
            "llm1_confirmed_cross_count": 7,
            "llm1_cv_confirmed_cross_count": 6,
            "llm1_cv_uncertain_retained_count": 1,
            "llm1_cv_high_score_retained_count": 2,
            "llm1_fallback_cross_count": 1,
            "llm1_fallback_verified_count": 1,
            "llm1_independent_scan_count": 5,
            "llm1_independent_supported_count": 4,
            "llm1_rejected_candidate_count": 5,
            "llm1_uncertain_candidate_count": 0,
            "llm1_candidate_audit_valid": True,
            "llm1_local_geometry_merge_count": 2,
            "llm1_truth_matched_count": 6,
            "llm1_truth_recall": 1.0,
            "llm1_model_confirmed_truth_matched_count": 4,
            "llm1_model_confirmed_truth_recall": 0.666667,
            "llm1_false_cross_count": 1,
            "llm1_duplicate_truth_candidate_count": 2,
            "llm2_matched_question_count": 6,
            "llm2_unmatched_cross_count": 1,
            "llm2_assignment_audit_valid": True,
            "llm2_retry_trigger_count": 2,
            "llm2_retry_request_count": 2,
            "llm2_retry_suppressed_count": 1,
            "llm2_retry_accepted_count": 1,
            "llm2_retry_rejected_count": 1,
            "llm2_first_pass_true_anchor_localization_failure_count": 2,
            "llm2_first_pass_false_anchor_matched_count": 1,
            "geometry_violation_count": 1,
            "duplicate_question_candidate_count": 2,
            "duplicate_truth_candidate_count": 1,
            "truth_matched_count": 6,
            "truth_count": 6,
            "truth_recall": 1.0,
            "content_ocr_status": "not_run",
        },
    )

    diagnostic._write_report(tmp_path, [summary])

    report = (tmp_path / "comparison-report.md").read_text("utf-8")
    assert "新方案CV候选" in report
    assert "新方案确认红叉" in report
    assert "保留uncertain" in report
    assert "保留高分CV" in report
    assert "LLM漏检补充" in report
    assert "复核通过fallback" in report
    assert "独立扫描红叉" in report
    assert "独立扫描支持" in report
    assert "本地几何合并" in report
    assert "LLM1确认真值召回" in report
    assert "保留锚点真值召回" in report
    assert "LLM1区域外红叉" in report
    assert "LLM1真值重复" in report
    assert "新方案错题定位" in report
    assert "重复错题候选" in report
    assert "真值重复归属" in report
    assert "新方案真值召回" in report
    assert "内容/OCR状态" in report
    assert "召回优先新方案判定" in report
    assert "保留LLM1拒绝CV" in report
    assert "稳定错题事件" in report
    assert "稳定真值召回" in report
    assert "新方案LLM请求" in report
    assert "定向复查新增找回" in report
    assert "定向复查新增误报" in report
    assert "定向复查触发" in report
    assert "定向复查请求" in report
    assert "复查触发抑制" in report
    assert "复查结果接纳" in report
    assert "复查结果拒绝" in report
    assert "LLM2真锚定位失败" in report
    assert "稳定最小真值覆盖" in report
    assert "fallback生成锚点" in report


def test_timing_report_compares_old_and_new_flows_with_stage_costs(tmp_path):
    diagnostic = _load_script_module()
    summary = diagnostic.build_summary(
        label="page35",
        expected_count=5,
        cv={"raw_component_count": 18, "evidence_group_count": 8},
        pipeline=None,
        cross_anchor_experiment={
            "llm1_verification_run_count": 3,
            "llm2_retry_trigger_count": 2,
            "llm2_retry_request_count": 2,
            "llm_request_count": 8,
            "timings_ms": {
                "llm1_candidate_verification": 1200.0,
                "independent_cross_scan": 300.0,
                "fallback_montage_and_verification": 400.0,
                "llm2_localization": 900.0,
                "llm2_localization_run_002": 450.0,
                "post_llm2_audit": 10.0,
            },
        },
    )
    summary["timings_ms"] = {
        "total": 5000.0,
        "red_evidence_cv": 20.0,
        "cross_candidate_cv": 30.0,
        "production_pipeline": 1000.0,
        "stable_event_experiment": 1100.0,
        "cross_anchor_experiment": 2800.0,
    }

    diagnostic._write_timing_report(tmp_path, [summary])

    report = (tmp_path / "timing-report.md").read_text("utf-8")
    assert "旧生产流程" in report
    assert "旧stable实验" in report
    assert "新方案总耗时" in report
    assert "定向复查耗时" in report
    assert "定向复查请求" in report
    assert "| page35 | 5000.0 | 20.0 | 30.0 | 1000.0 | 1100.0 | 2800.0 | 3 | 8 |" in report


def test_main_compare_cross_anchor_loads_inputs_and_forwards_experiment_flag(
    tmp_path,
    monkeypatch,
):
    diagnostic = _load_script_module()
    source = tmp_path / "page.jpg"
    Image.new("RGB", (20, 20), "white").save(source)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "analysis_max_edge": 1000,
                "red_min_channel": 60,
                "red_min_excess": 6,
                "axis_line_min_density": 0.65,
                "arm_inner_radius_ratio": 0.006,
                "arm_outer_radius_ratio": 0.03,
                "diagonal_band_ratio": 0.3,
                "arm_min_density": 0.11,
                "center_radius_ratio": 0.004,
                "center_min_density": 0.06,
                "candidate_merge_radius_ratio": 0.04,
                "bbox_padding_ratio": 0.01,
                    "truth_match_margin_ratio": 0.01,
                    "cross_anchor_question_max_area_ratio": 0.3,
                    "cross_anchor_question_max_gap_ratio": 0.04,
                    "cross_anchor_duplicate_question_iou_threshold": 0.2,
                    "cross_anchor_llm1_verification_runs": 1,
                    "cross_anchor_llm2_localization_runs": 2,
                    "cross_anchor_llm2_retry_min_center_gap_ratio": 0.03,
                    "cross_anchor_llm2_retry_crop_padding_ratio": 0.12,
                    "cross_anchor_llm2_retry_first_pass_min_question_cross_area_ratio": 1.1,
                    "cross_anchor_llm2_retry_min_question_cross_area_ratio": 1.5,
                    "cross_anchor_llm2_retry_max_question_cross_iou": 0.8,
                    "cross_anchor_llm2_retry_shared_question_iou_threshold": 0.8,
                    "cross_anchor_llm2_retry_shared_question_min_anchor_distance_ratio": 0.1,
                    "cross_anchor_llm2_retry_shared_question_min_group_size": 3,
                    "cross_anchor_llm2_retry_max_requests_per_page": 1,
                    "cross_anchor_fallback_generates_anchors": False,
                    "cross_anchor_retain_uncertain_candidates": True,
                    "cross_anchor_retain_rejected_candidates": True,
                    "cross_anchor_retain_uncertain_fallback_candidates": True,
                    "cross_anchor_high_cv_min_arm_density": 0.25,
                    "cross_anchor_high_cv_min_center_density": 0.7,
                    "cross_anchor_cv_dedupe_iou_threshold": 0.5,
                    "cross_anchor_cv_dedupe_center_distance_ratio": 0.015,
                    "cross_anchor_fallback_merge_iou_threshold": 0.2,
                    "cross_anchor_fallback_merge_center_distance_ratio": 0.04,
                    "cross_anchor_montage_full_page_max_edge": 1400,
                    "cross_anchor_montage_tile_edge": 320,
                        "cross_anchor_montage_columns": 3,
                        "cross_anchor_montage_crop_padding_ratio": 0.03,
                        "cross_anchor_llm2_batch_size": 3,
                        "question_truth_min_iou": 0.2,
            }
        ),
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "pages": {
                    "sample": {
                        "regions": [
                            {
                                "truth_id": "T1",
                                "source_bbox_normalized": [0.1, 0.1, 0.3, 0.3],
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    received = []

    def fake_run_case(**kwargs):
        received.append(kwargs)
        summary = diagnostic.build_summary(
            label=kwargs["label"],
            expected_count=1,
            cv={"raw_component_count": 0, "evidence_group_count": 0},
            pipeline=None,
        )
        summary.update(
            {
                "error": None,
                "stable_experiment_error": None,
                "cross_anchor_experiment_error": None,
            }
        )
        return summary

    monkeypatch.setattr(diagnostic, "run_case", fake_run_case)
    monkeypatch.setattr(
        diagnostic.sys,
        "argv",
        [
            "diagnose_vision_pipeline.py",
            f"sample={source}",
            "--expected",
            "sample=1",
            "--compare-cross-anchor",
            "--cross-cv-config",
            str(config_path),
            "--truth-regions",
            str(truth_path),
            "--output",
            str(output),
        ],
    )

    assert diagnostic.main() == 0
    assert received[0]["compare_cross_anchor"] is True
    assert received[0]["cross_cv_config"]["question_truth_min_iou"] == 0.2
    assert received[0]["truth_regions"][0]["truth_id"] == "T1"
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    assert manifest["compare_cross_anchor"] is True


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
        primitive_duplicate_containment_threshold=0.8,
        cross_circle_max_center_distance=0.2,
    )

    assert [call[0] for call in calls] == ["detect", "consolidate"]
    assert result["event_count"] == 1
    assert result["primitive_membership_violations"] == []
    assert result["duplicate_primitive_candidates"] == []
    assert result["cross_circle_geometry_candidates"] == []
    assert result["cv_event_support"][0]["status"] == "supported"
    assert (case_dir / "stable-event-experiment" / "stable-events.json").is_file()
    assert (
        case_dir / "stable-event-experiment" / "cv-post-validation.json"
    ).is_file()
    assert (
        case_dir
        / "stable-event-experiment"
        / "duplicate-primitive-candidates.json"
    ).is_file()
    assert (
        case_dir / "stable-event-experiment" / "cross-circle-geometry-audit.json"
    ).is_file()
