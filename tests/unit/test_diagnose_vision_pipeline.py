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
    assert truth["page33"][0]["truth_id"] == "T1"
    with pytest.raises(ValueError, match="missing truth page: page34"):
        diagnostic.load_cross_cv_inputs(
            config_path,
            truth_path,
            ["page33", "page34"],
        )


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

    with pytest.raises(ValueError, match="cross_anchor_question_max_area_ratio"):
        diagnostic.load_cross_cv_inputs(config_path, truth_path, ["page33"])


def test_cross_candidate_disposition_audit_requires_every_cv_candidate_once():
    diagnostic = _load_script_module()
    result = diagnostic.CrossCandidateVerificationResult.model_validate(
        {
            "confirmed_crosses": [
                {
                    "source_candidate_ids": [0, 1],
                    "bbox": [0.1, 0.1, 0.3, 0.3],
                    "confidence": 0.95,
                }
            ],
            "rejected_candidate_ids": [1],
            "uncertain_candidate_ids": [],
            "additional_crosses": [],
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


def test_confirmed_cross_anchors_keep_cv_provenance_and_llm_fallback():
    diagnostic = _load_script_module()
    result = diagnostic.CrossCandidateVerificationResult.model_validate(
        {
            "confirmed_crosses": [
                {
                    "source_candidate_ids": [3, 2],
                    "bbox": [0.5, 0.5, 0.6, 0.6],
                    "confidence": 0.9,
                }
            ],
            "rejected_candidate_ids": [0],
            "uncertain_candidate_ids": [1],
            "additional_crosses": [
                {"bbox": [0.1, 0.1, 0.2, 0.2], "confidence": 0.8}
            ],
        }
    )

    anchors = diagnostic.build_confirmed_cross_anchors(result)

    assert anchors == [
        {
            "cross_id": 0,
            "source": "llm_fallback",
            "source_candidate_ids": [],
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "confidence": 0.8,
        },
        {
            "cross_id": 1,
            "source": "cv_confirmed",
            "source_candidate_ids": [2, 3],
            "bbox": [0.5, 0.5, 0.6, 0.6],
            "confidence": 0.9,
        },
    ]


def test_cross_anchor_assignment_audit_preserves_missing_and_unknown_ids():
    diagnostic = _load_script_module()
    result = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": False,
                    "question_bbox": None,
                    "answer_bbox": None,
                    "prompt_bbox": None,
                    "raw_text": None,
                    "instruction": None,
                    "prompt_text": None,
                    "normalized_text": None,
                    "answer": None,
                    "subject": None,
                    "question_type": None,
                    "tags": [],
                    "difficulty": None,
                    "uncertain_segments": [],
                    "auxiliary_circle_bboxes": [],
                    "teacher_annotation": None,
                    "confidence": 0.4,
                },
                {
                    "cross_id": 9,
                    "matched": False,
                    "question_bbox": None,
                    "answer_bbox": None,
                    "prompt_bbox": None,
                    "raw_text": None,
                    "instruction": None,
                    "prompt_text": None,
                    "normalized_text": None,
                    "answer": None,
                    "subject": None,
                    "question_type": None,
                    "tags": [],
                    "difficulty": None,
                    "uncertain_segments": [],
                    "auxiliary_circle_bboxes": [],
                    "teacher_annotation": None,
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
                    "answer_bbox": [0.2, 0.2, 0.3, 0.3],
                    "prompt_bbox": [0.1, 0.1, 0.3, 0.2],
                    "raw_text": "错字",
                    "instruction": "看拼音写词语",
                    "prompt_text": "cuo zi",
                    "normalized_text": "错字",
                    "answer": "错字",
                    "subject": "chinese",
                    "question_type": "write_word",
                    "tags": ["词语"],
                    "difficulty": 2,
                    "uncertain_segments": [],
                    "auxiliary_circle_bboxes": [],
                    "teacher_annotation": None,
                    "confidence": 0.9,
                }
            ]
        }
    )

    audit = diagnostic.audit_anchored_question_geometry(
        result,
        anchors,
        max_area_ratio=0.3,
    )

    assert audit["valid"] is False
    assert audit["violations"] == [
        {
            "cross_id": 0,
            "reasons": ["question_bbox_does_not_contain_cross_center"],
            "question_area_ratio": 0.09,
        }
    ]


def test_anchored_question_truth_comparison_uses_question_bbox_iou():
    diagnostic = _load_script_module()
    result = diagnostic.CrossAnchoredQuestionResult.model_validate(
        {
            "items": [
                {
                    "cross_id": 0,
                    "matched": True,
                    "question_bbox": [0.1, 0.1, 0.4, 0.4],
                    "answer_bbox": [0.2, 0.2, 0.3, 0.3],
                    "prompt_bbox": [0.1, 0.1, 0.3, 0.2],
                    "raw_text": "错字",
                    "instruction": "看拼音写词语",
                    "prompt_text": "cuo zi",
                    "normalized_text": "错字",
                    "answer": "错字",
                    "subject": "chinese",
                    "question_type": "write_word",
                    "tags": [],
                    "difficulty": 2,
                    "uncertain_segments": [],
                    "auxiliary_circle_bboxes": [],
                    "teacher_annotation": None,
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
    assert comparison["items"][0] == {
        "cross_id": 0,
        "truth_id": "T1",
        "best_iou": 1.0,
        "meets_threshold": True,
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


def test_cross_anchor_experiment_runs_candidate_verification_localization_and_ocr(
    tmp_path,
):
    diagnostic = _load_script_module()
    from app.services.local_ocr_verification import OCRVerification

    source = tmp_path / "page.jpg"
    Image.new("RGB", (200, 200), "white").save(source)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    candidate_overlay = tmp_path / "candidate-overlay.jpg"
    Image.new("RGB", (200, 200), "white").save(candidate_overlay)
    candidates = [
        {"candidate_id": 0, "bbox": [0.1, 0.1, 0.2, 0.2], "center": [0.15, 0.15]},
        {"candidate_id": 1, "bbox": [0.8, 0.8, 0.9, 0.9], "center": [0.85, 0.85]},
    ]
    calls = []

    class FakeClient:
        def verify_cross_candidates(self, image_path, received_candidates):
            calls.append(("verify", Path(image_path).name, len(received_candidates)))
            return diagnostic.CrossCandidateVerificationResult.model_validate(
                {
                    "confirmed_crosses": [
                        {
                            "source_candidate_ids": [0],
                            "bbox": [0.1, 0.1, 0.2, 0.2],
                            "confidence": 0.95,
                        }
                    ],
                    "rejected_candidate_ids": [1],
                    "uncertain_candidate_ids": [],
                    "additional_crosses": [
                        {"bbox": [0.6, 0.6, 0.7, 0.7], "confidence": 0.85}
                    ],
                }
            )

        def locate_cross_anchored_questions(self, image_path, anchors, subject_hint):
            calls.append(("localize", Path(image_path).name, len(anchors), subject_hint))
            return diagnostic.CrossAnchoredQuestionResult.model_validate(
                {
                    "items": [
                        {
                            "cross_id": 0,
                            "matched": True,
                            "question_bbox": [0.05, 0.05, 0.3, 0.3],
                            "answer_bbox": [0.1, 0.1, 0.25, 0.25],
                            "prompt_bbox": [0.05, 0.05, 0.25, 0.12],
                            "raw_text": "甲",
                            "instruction": "看拼音写词语",
                            "prompt_text": "jia",
                            "normalized_text": "甲",
                            "answer": "甲",
                            "subject": "chinese",
                            "question_type": "write_word",
                            "tags": ["词语"],
                            "difficulty": 1,
                            "uncertain_segments": [],
                            "auxiliary_circle_bboxes": [],
                            "teacher_annotation": None,
                            "confidence": 0.9,
                        },
                        {
                            "cross_id": 1,
                            "matched": True,
                            "question_bbox": [0.5, 0.5, 0.75, 0.75],
                            "answer_bbox": [0.58, 0.58, 0.72, 0.72],
                            "prompt_bbox": [0.5, 0.5, 0.72, 0.58],
                            "raw_text": "乙",
                            "instruction": "看拼音写词语",
                            "prompt_text": "yi",
                            "normalized_text": "乙",
                            "answer": "乙",
                            "subject": "chinese",
                            "question_type": "write_word",
                            "tags": ["词语"],
                            "difficulty": 1,
                            "uncertain_segments": [],
                            "auxiliary_circle_bboxes": [],
                            "teacher_annotation": None,
                            "confidence": 0.9,
                        },
                    ]
                }
            )

    class FakeOCR:
        def verify_crop(self, image_path, bbox, target_index, items):
            calls.append(("ocr", target_index, len(items), bbox))
            return OCRVerification(
                status="support",
                matched_index=target_index,
                text_summary=items[target_index].prompt_text,
                confidence=0.98,
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
            "question_truth_min_iou": 0.2,
        },
        subject_hint="chinese",
        ocr_verifier=FakeOCR(),
    )

    assert [call[0] for call in calls] == ["verify", "localize", "ocr", "ocr"]
    assert summary == {
        "cv_candidate_count": 2,
        "llm1_confirmed_cross_count": 2,
        "llm1_cv_confirmed_cross_count": 1,
        "llm1_fallback_cross_count": 1,
        "llm1_rejected_candidate_count": 1,
        "llm1_uncertain_candidate_count": 0,
        "llm1_candidate_audit_valid": True,
        "llm1_truth_matched_count": 2,
        "llm1_truth_recall": 1.0,
        "llm1_false_cross_count": 0,
        "llm2_matched_question_count": 2,
        "llm2_unmatched_cross_count": 0,
        "llm2_assignment_audit_valid": True,
        "geometry_violation_count": 0,
        "truth_matched_count": 2,
        "truth_count": 2,
        "truth_recall": 1.0,
        "ocr_status_counts": {"support": 2},
    }
    experiment_dir = case_dir / "cross-anchor-experiment"
    assert (experiment_dir / "llm1-candidate-verification.json").is_file()
    assert (experiment_dir / "llm1-candidate-membership-audit.json").is_file()
    assert (experiment_dir / "confirmed-crosses.json").is_file()
    assert (experiment_dir / "confirmed-crosses-overlay.jpg").is_file()
    assert (experiment_dir / "llm1-truth-comparison.json").is_file()
    assert (experiment_dir / "llm2-anchored-questions.json").is_file()
    assert (experiment_dir / "llm2-cross-assignment-audit.json").is_file()
    assert (experiment_dir / "question-geometry-audit.json").is_file()
    assert (experiment_dir / "truth-comparison.json").is_file()
    assert (experiment_dir / "ocr-audit.json").is_file()
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
                {
                    "confirmed_crosses": [],
                    "rejected_candidate_ids": [],
                    "uncertain_candidate_ids": [],
                    "additional_crosses": [],
                }
            )

        def locate_cross_anchored_questions(self, *_args):
            raise AssertionError("LLM2 must not run after invalid LLM1 membership")

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
                }
            ],
            candidate_overlay_path=overlay,
            truth_regions=[],
            config={
                "cross_anchor_question_max_area_ratio": 0.3,
                "question_truth_min_iou": 0.2,
            },
            subject_hint="chinese",
            ocr_verifier=None,
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
                    "confirmed_crosses": [],
                    "rejected_candidate_ids": [0],
                    "uncertain_candidate_ids": [],
                    "additional_crosses": [],
                }
            )

        def locate_cross_anchored_questions(self, *_args):
            raise AssertionError("LLM2 must not run without a confirmed cross")

    summary = diagnostic.run_cross_anchor_experiment(
        image_path=source,
        case_dir=tmp_path / "case",
        client=FakeClient(),
        cv_candidates=[
            {
                "candidate_id": 0,
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "center": [0.15, 0.15],
            }
        ],
        candidate_overlay_path=overlay,
        truth_regions=[
            {"truth_id": "T1", "source_bbox_normalized": [0.1, 0.1, 0.3, 0.3]}
        ],
        config={
            "cross_anchor_question_max_area_ratio": 0.3,
            "question_truth_min_iou": 0.2,
        },
        subject_hint="chinese",
        ocr_verifier=None,
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
            "llm1_fallback_cross_count": 1,
            "llm1_rejected_candidate_count": 5,
            "llm1_uncertain_candidate_count": 0,
            "llm1_candidate_audit_valid": True,
            "llm1_truth_matched_count": 6,
            "llm1_truth_recall": 1.0,
            "llm1_false_cross_count": 1,
            "llm2_matched_question_count": 6,
            "llm2_unmatched_cross_count": 1,
            "llm2_assignment_audit_valid": True,
            "geometry_violation_count": 1,
            "truth_matched_count": 6,
            "truth_count": 6,
            "truth_recall": 1.0,
            "ocr_status_counts": {"support": 4, "inconclusive": 2},
        },
    )

    checkpoints = summary["checkpoints"]
    assert summary["cross_anchor_experiment_status"] == "completed"
    assert checkpoints["pipeline_truth_matched_count"] == 4
    assert checkpoints["pipeline_truth_recall"] == 0.666667
    assert checkpoints["cross_anchor_cv_candidate_count"] == 12
    assert checkpoints["cross_anchor_confirmed_cross_count"] == 7
    assert checkpoints["cross_anchor_fallback_cross_count"] == 1
    assert checkpoints["cross_anchor_llm1_truth_recall"] == 1.0
    assert checkpoints["cross_anchor_llm1_false_cross_count"] == 1
    assert checkpoints["cross_anchor_matched_question_count"] == 6
    assert checkpoints["cross_anchor_geometry_violation_count"] == 1
    assert checkpoints["cross_anchor_truth_recall"] == 1.0
    assert checkpoints["cross_anchor_ocr_contradiction_count"] == 0


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
            "llm1_fallback_cross_count": 1,
            "llm1_rejected_candidate_count": 5,
            "llm1_uncertain_candidate_count": 0,
            "llm1_candidate_audit_valid": True,
            "llm1_truth_matched_count": 6,
            "llm1_truth_recall": 1.0,
            "llm1_false_cross_count": 1,
            "llm2_matched_question_count": 6,
            "llm2_unmatched_cross_count": 1,
            "llm2_assignment_audit_valid": True,
            "geometry_violation_count": 1,
            "truth_matched_count": 6,
            "truth_count": 6,
            "truth_recall": 1.0,
            "ocr_status_counts": {"wrong_candidate": 1, "support": 5},
        },
    )

    diagnostic._write_report(tmp_path, [summary])

    report = (tmp_path / "comparison-report.md").read_text("utf-8")
    assert "新方案CV候选" in report
    assert "新方案确认红叉" in report
    assert "LLM漏检补充" in report
    assert "LLM1真值召回" in report
    assert "LLM1区域外红叉" in report
    assert "新方案错题定位" in report
    assert "新方案真值召回" in report
    assert "OCR矛盾" in report


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
