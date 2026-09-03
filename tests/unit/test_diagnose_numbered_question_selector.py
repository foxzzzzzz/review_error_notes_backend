import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw


BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "diagnose_numbered_question_selector.py"
CONFIG_PATH = BACKEND_ROOT / "scripts" / "numbered_question_selector_config.json"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "diagnose_numbered_question_selector", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    return {
        "ocr_line_confidence_threshold": 0.45,
        "ocr_line_min_width_ratio": 0.04,
        "ocr_line_max_width_ratio": 0.5,
        "ocr_line_min_height_ratio": 0.015,
        "ocr_line_max_height_ratio": 0.1,
        "layout_red_min_channel": 100,
        "layout_red_min_excess": 4,
        "layout_horizontal_kernel_width_ratio": 0.04,
        "layout_vertical_kernel_height_ratio": 0.02,
        "layout_dilation_pixels": 3,
        "layout_component_min_width_ratio": 0.07,
        "layout_component_max_width_ratio": 0.7,
        "layout_component_min_height_ratio": 0.02,
        "layout_component_max_height_ratio": 0.2,
        "layout_typical_max_width_ratio": 0.3,
        "layout_typical_max_height_ratio": 0.09,
        "layout_split_width_multiplier": 1.5,
        "layout_split_height_multiplier": 1.6,
        "fallback_question_width_ratio": 0.2,
        "fallback_question_height_ratio": 0.055,
        "question_width_cap_multiplier": 1.3,
        "question_horizontal_half_width_multipliers": [0.56, 0.78],
        "question_height_multiplier": 2.8,
        "candidate_top_k_per_anchor": 3,
        "candidate_dedup_iou_threshold": 0.9,
        "context_padding_x_ratio": 0.03,
        "context_padding_y_ratio": 0.03,
        "truth_anchor_margin_ratio": 0.01,
        "truth_match_min_iou": 0.2,
        "truth_match_min_coverage": 0.75,
        "sibling_intrusion_min_coverage": 0.1,
        "montage_columns": 3,
        "montage_tile_width": 320,
        "montage_tile_height": 240,
        "montage_label_height": 28,
        "montage_jpeg_quality": 90,
        "anchor_montage_columns": 3,
        "anchor_montage_tile_width": 320,
        "anchor_montage_tile_height": 240,
        "anchor_montage_label_height": 28,
        "anchor_montage_crop_padding_ratio": 0.02,
        "local_recheck_red_min_channel": 100,
        "local_recheck_red_min_excess": 20,
        "local_recheck_center_window_ratio": 0.08,
        "local_recheck_min_center_red_ratio": 0.05,
        "local_recheck_bottom_page_number_min_y_ratio": 0.9,
        "local_recheck_page_number_max_distance_ratio": 0.04,
        "atomic_duplicate_iou_threshold": 0.45,
        "atomic_duplicate_ocr_jaccard_threshold": 0.5,
        "atomic_same_row_center_y_ratio": 0.45,
        "atomic_same_column_center_x_ratio": 0.45,
        "atomic_horizontal_min_center_separation_ratio": 0.25,
        "atomic_horizontal_partition_overlap_ratio": 0.03,
        "atomic_vertical_min_center_separation_ratio": 0.5,
        "atomic_vertical_clip_min_width_multiplier": 1.15,
        "atomic_ocr_row_merge_gap_height_multiplier": 0.7,
    }


def test_config_documents_every_runtime_parameter():
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert set(payload) - {"_descriptions"} == set(payload["_descriptions"])


def test_build_candidates_is_deterministic_and_keeps_context_separate(tmp_path):
    diagnostic = _load_script_module()
    image_path = tmp_path / "page.jpg"
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 60, 100, 100), outline=(180, 90, 90), width=2)
    draw.rectangle((110, 60, 180, 100), outline=(180, 90, 90), width=2)
    image.save(image_path)
    anchors = [{"cross_id": 7, "bbox": [0.35, 0.3, 0.45, 0.4]}]
    ocr_lines = [
        {"bbox": [0.2, 0.3, 0.45, 0.38], "confidence": 0.9},
        {"bbox": [0.58, 0.3, 0.8, 0.38], "confidence": 0.9},
    ]

    first = diagnostic.build_numbered_candidates(
        image_path=image_path,
        anchors=anchors,
        ocr_lines=ocr_lines,
        config=_config(),
    )
    second = diagnostic.build_numbered_candidates(
        image_path=image_path,
        anchors=anchors,
        ocr_lines=ocr_lines,
        config=_config(),
    )

    assert first == second
    assert first["anchor_candidates"][7]
    assert len(first["anchor_candidates"][7]) <= 3
    assert {candidate["boundary_variant"] for candidate in first["candidates"]} == {
        "compact",
        "standard",
    }
    for candidate in first["candidates"]:
        assert candidate["candidate_id"].startswith("Q")
        assert candidate["question_bbox"] != candidate["context_bbox"]
        assert diagnostic.bbox_contains(
            candidate["context_bbox"], candidate["question_bbox"]
        )


def test_selection_audit_rejects_unknown_and_cross_anchor_candidate_ids():
    diagnostic = _load_script_module()
    allowed = {0: ["Q0", "Q1"], 1: ["Q2"]}
    selections = [
        {
            "cross_id": 0,
            "decision": "selected",
            "selected_candidate_id": "Q2",
            "confidence": 0.9,
        },
        {
            "cross_id": 1,
            "decision": "selected",
            "selected_candidate_id": "Q9",
            "confidence": 0.8,
        },
    ]

    audit = diagnostic.audit_selections(selections, allowed)

    assert audit["valid"] is False
    assert audit["accepted"] == []
    assert {item["reason"] for item in audit["violations"]} == {
        "candidate_not_allowed_for_anchor",
        "unknown_candidate_id",
    }


def test_selection_audit_requires_each_anchor_once_and_normalizes_none():
    diagnostic = _load_script_module()
    allowed = {0: ["Q0"], 1: ["Q1"]}
    selections = [
        {
            "cross_id": 0,
            "decision": "none",
            "selected_candidate_id": None,
            "confidence": 0.8,
        }
    ]

    audit = diagnostic.audit_selections(selections, allowed)

    assert audit["valid"] is False
    assert audit["accepted"] == [
        {
            "cross_id": 0,
            "decision": "none",
            "selected_candidate_id": None,
            "confidence": 0.8,
        }
    ]
    assert audit["violations"] == [{"cross_id": 1, "reason": "missing_anchor"}]


def test_selected_candidates_are_deduplicated_without_changing_geometry():
    diagnostic = _load_script_module()
    candidates = [
        {
            "candidate_id": "Q0",
            "question_bbox": [0.1, 0.2, 0.4, 0.5],
            "context_bbox": [0.05, 0.15, 0.45, 0.55],
        }
    ]
    accepted = [
        {
            "cross_id": 2,
            "decision": "selected",
            "selected_candidate_id": "Q0",
            "confidence": 0.8,
        },
        {
            "cross_id": 3,
            "decision": "selected",
            "selected_candidate_id": "Q0",
            "confidence": 0.9,
        },
    ]

    events = diagnostic.build_selected_events(accepted, candidates)

    assert events == [
        {
            "event_id": 0,
            "candidate_id": "Q0",
            "cross_ids": [2, 3],
            "question_bbox": [0.1, 0.2, 0.4, 0.5],
            "confidence": 0.9,
        }
    ]


def test_candidate_oracle_events_preserve_allowed_anchor_membership():
    diagnostic = _load_script_module()
    candidates = [
        {
            "candidate_id": "Q0",
            "question_bbox": [0.1, 0.2, 0.4, 0.5],
            "context_bbox": [0.05, 0.15, 0.45, 0.55],
            "cross_ids": [3, 2],
        }
    ]

    events = diagnostic.build_candidate_oracle_events(candidates)

    assert events == [
        {
            "event_id": 0,
            "candidate_id": "Q0",
            "cross_ids": [2, 3],
            "question_bbox": [0.1, 0.2, 0.4, 0.5],
            "confidence": 1.0,
        }
    ]


def test_truth_comparison_does_not_credit_false_anchor_geometry():
    diagnostic = _load_script_module()
    truth = [
        {"truth_id": "T1", "source_bbox_normalized": [0.1, 0.1, 0.4, 0.4]}
    ]
    anchors = [{"cross_id": 9, "bbox": [0.8, 0.8, 0.9, 0.9]}]
    events = [
        {
            "event_id": 0,
            "candidate_id": "Q0",
            "cross_ids": [9],
            "question_bbox": [0.1, 0.1, 0.4, 0.4],
            "confidence": 0.9,
        }
    ]

    comparison = diagnostic.compare_selected_events_to_truth(
        anchors=anchors,
        events=events,
        truth_regions=truth,
        config=_config(),
    )

    assert comparison["matched_truth_ids"] == []
    assert comparison["truth_recall"] == 0.0
    assert comparison["false_event_ids"] == [0]


def test_selector_prompt_forbids_coordinates_and_allows_circle_as_auxiliary():
    diagnostic = _load_script_module()

    assert "不得返回或生成任何 bbox" in diagnostic.NUMBERED_SELECTION_PROMPT
    assert "红圈" in diagnostic.NUMBERED_SELECTION_PROMPT
    assert "selected_candidate_id" in diagnostic.NUMBERED_SELECTION_PROMPT


def test_local_recheck_rejects_empty_center_but_preserves_centered_cross(tmp_path):
    diagnostic = _load_script_module()
    image_path = tmp_path / "anchors.jpg"
    image = Image.new("RGB", (300, 150), "white")
    draw = ImageDraw.Draw(image)
    draw.line((35, 35, 115, 115), fill=(220, 20, 20), width=8)
    draw.line((115, 35, 35, 115), fill=(220, 20, 20), width=8)
    draw.ellipse((185, 35, 265, 115), outline=(220, 20, 20), width=8)
    image.save(image_path)
    anchors = [
        {"cross_id": 0, "bbox": [0.0, 0.0, 0.5, 1.0]},
        {"cross_id": 1, "bbox": [0.5, 0.0, 1.0, 1.0]},
    ]

    assessments = diagnostic.assess_local_anchor_geometry(
        image_path=image_path,
        anchors=anchors,
        ocr_lines=[],
        config=_config(),
    )

    assert assessments == [
        {
            "cross_id": 0,
            "decision": "keep",
            "reason": "center_red_supported",
            "center_red_ratio": assessments[0]["center_red_ratio"],
            "page_number_text": None,
        },
        {
            "cross_id": 1,
            "decision": "reject",
            "reason": "insufficient_center_red",
            "center_red_ratio": 0.0,
            "page_number_text": None,
        },
    ]


def test_local_recheck_rejects_bottom_numeric_page_marker(tmp_path):
    diagnostic = _load_script_module()
    image_path = tmp_path / "page.jpg"
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.line((85, 175, 115, 195), fill=(220, 20, 20), width=6)
    draw.line((115, 175, 85, 195), fill=(220, 20, 20), width=6)
    image.save(image_path)
    anchors = [{"cross_id": 3, "bbox": [0.4, 0.84, 0.6, 1.0]}]
    ocr_lines = [{"text": "35", "bbox": [0.46, 0.91, 0.54, 0.97], "confidence": 0.99}]

    assessments = diagnostic.assess_local_anchor_geometry(
        image_path=image_path,
        anchors=anchors,
        ocr_lines=ocr_lines,
        config=_config(),
    )

    assert assessments[0]["decision"] == "reject"
    assert assessments[0]["reason"] == "bottom_page_number"
    assert assessments[0]["page_number_text"] == "35"


def test_consensus_filter_rejects_only_when_local_and_llm_agree():
    diagnostic = _load_script_module()
    assessments = [
        {"cross_id": 0, "decision": "reject"},
        {"cross_id": 1, "decision": "reject"},
        {"cross_id": 2, "decision": "keep"},
    ]
    verifications = [
        {"cross_id": 0, "decision": "not_cross"},
        {"cross_id": 1, "decision": "uncertain"},
        {"cross_id": 2, "decision": "not_cross"},
    ]

    result = diagnostic.consensus_anchor_filter(
        anchor_ids=[0, 1, 2],
        local_assessments=assessments,
        llm_verifications=verifications,
    )

    assert result == {
        "kept_cross_ids": [1, 2],
        "rejected_cross_ids": [0],
    }


def test_deterministic_events_prefer_standard_candidate_and_deduplicate():
    diagnostic = _load_script_module()
    anchors = [
        {"cross_id": 0, "confidence": 0.8},
        {"cross_id": 1, "confidence": 0.9},
    ]
    candidates = [
        {
            "candidate_id": "Q0",
            "boundary_variant": "compact",
            "question_bbox": [0.2, 0.2, 0.4, 0.4],
        },
        {
            "candidate_id": "Q1",
            "boundary_variant": "standard",
            "question_bbox": [0.1, 0.1, 0.5, 0.5],
        },
    ]
    allowed = {0: ["Q0", "Q1"], 1: ["Q0", "Q1"]}

    events = diagnostic.build_deterministic_events(
        anchors=anchors,
        candidates=candidates,
        allowed=allowed,
        kept_cross_ids=[0, 1],
    )

    assert events == [
        {
            "event_id": 0,
            "candidate_id": "Q1",
            "cross_ids": [0, 1],
            "question_bbox": [0.1, 0.1, 0.5, 0.5],
            "confidence": 0.9,
        }
    ]


def test_anchor_verification_audit_rejects_inconsistent_visual_evidence():
    diagnostic = _load_script_module()
    verifications = [
        {
            "cross_id": 0,
            "decision": "real_cross",
            "visual_evidence": "circle_or_oval",
            "confidence": 0.9,
        },
        {
            "cross_id": 1,
            "decision": "not_cross",
            "visual_evidence": "printed_grid_or_text",
            "confidence": 0.8,
        },
    ]

    audit = diagnostic.audit_anchor_verifications(verifications, [0, 1])

    assert audit["valid"] is False
    assert audit["accepted"] == [verifications[1]]
    assert audit["violations"] == [
        {"cross_id": 0, "reason": "real_cross_without_cross_evidence"}
    ]


def test_anchor_montage_has_one_labeled_tile_per_anchor(tmp_path):
    diagnostic = _load_script_module()
    image_path = tmp_path / "page.jpg"
    output_path = tmp_path / "montage.jpg"
    Image.new("RGB", (200, 200), "white").save(image_path)
    anchors = [
        {"cross_id": 2, "bbox": [0.1, 0.1, 0.3, 0.3]},
        {"cross_id": 7, "bbox": [0.6, 0.6, 0.8, 0.8]},
    ]

    diagnostic.write_anchor_verification_montage(
        image_path=image_path,
        output_path=output_path,
        anchors=anchors,
        config=_config(),
    )

    with Image.open(output_path) as montage:
        assert montage.size == (960, 240)


def test_anchor_prompt_does_not_ask_model_to_select_question_candidates():
    diagnostic = _load_script_module()

    assert "禁止默认全部确认" in diagnostic.ANCHOR_VERIFICATION_PROMPT
    assert "two_intersecting_red_diagonal_strokes" in diagnostic.ANCHOR_VERIFICATION_PROMPT
    assert "selected_candidate_id" not in diagnostic.ANCHOR_VERIFICATION_PROMPT


def test_atomic_events_merge_only_overlapping_boxes_with_shared_ocr_lines():
    diagnostic = _load_script_module()
    events = [
        {
            "event_id": 0,
            "candidate_id": "Q1",
            "cross_ids": [0],
            "question_bbox": [0.67, 0.06, 1.0, 0.22],
            "confidence": 0.8,
        },
        {
            "event_id": 1,
            "candidate_id": "Q3",
            "cross_ids": [1],
            "question_bbox": [0.67, 0.11, 1.0, 0.27],
            "confidence": 0.9,
        },
        {
            "event_id": 2,
            "candidate_id": "Q4",
            "cross_ids": [2],
            "question_bbox": [0.1, 0.1, 0.3, 0.3],
            "confidence": 0.7,
        },
    ]
    ocr_lines = [
        {"text": "same question", "bbox": [0.72, 0.14, 0.92, 0.2], "confidence": 0.9},
        {"text": "other", "bbox": [0.12, 0.14, 0.25, 0.2], "confidence": 0.9},
    ]

    refined, audit = diagnostic.build_atomic_question_events(
        events=events,
        ocr_lines=ocr_lines,
        config=_config(),
    )

    assert refined[0] == {
        "event_id": 0,
        "candidate_id": "Q1",
        "cross_ids": [0, 1],
        "question_bbox": [0.67, 0.06, 1.0, 0.22],
        "confidence": 0.9,
    }
    assert refined[1]["cross_ids"] == [2]
    assert audit["ocr_duplicate_groups"] == [[0, 1]]


def test_atomic_events_partition_same_row_and_wide_lower_sibling_overlap():
    diagnostic = _load_script_module()
    events = [
        {
            "event_id": 0,
            "candidate_id": "left",
            "cross_ids": [0],
            "question_bbox": [0.0, 0.214, 0.321, 0.37],
            "confidence": 0.9,
        },
        {
            "event_id": 1,
            "candidate_id": "right",
            "cross_ids": [1],
            "question_bbox": [0.197, 0.165, 0.541, 0.321],
            "confidence": 0.9,
        },
        {
            "event_id": 2,
            "candidate_id": "upper",
            "cross_ids": [2],
            "question_bbox": [0.214, 0.476, 0.543, 0.631],
            "confidence": 0.9,
        },
        {
            "event_id": 3,
            "candidate_id": "wide-lower",
            "cross_ids": [3],
            "question_bbox": [0.284, 0.602, 0.71, 0.757],
            "confidence": 0.8,
        },
    ]

    refined, audit = diagnostic.build_atomic_question_events(
        events=events,
        ocr_lines=[],
        config=_config(),
    )

    assert refined[0]["question_bbox"][2] > refined[1]["question_bbox"][0]
    assert refined[3]["question_bbox"][1] == 0.631
    assert audit["partitioned_event_ids"] == [0, 1, 3]


def test_atomic_events_do_not_clip_normal_width_stacked_question():
    diagnostic = _load_script_module()
    events = [
        {
            "event_id": 0,
            "candidate_id": "upper",
            "cross_ids": [6],
            "question_bbox": [0.006, 0.505, 0.279, 0.658],
            "confidence": 0.8,
        },
        {
            "event_id": 1,
            "candidate_id": "target",
            "cross_ids": [7],
            "question_bbox": [0.075, 0.603, 0.339, 0.755],
            "confidence": 0.9,
        },
        {
            "event_id": 2,
            "candidate_id": "lower",
            "cross_ids": [8],
            "question_bbox": [0.031, 0.664, 0.375, 0.816],
            "confidence": 0.8,
        },
    ]

    refined, audit = diagnostic.build_atomic_question_events(
        events=events,
        ocr_lines=[],
        config=_config(),
    )

    assert refined[1]["question_bbox"] == [0.075, 0.603, 0.339, 0.755]
    assert 1 not in audit["partitioned_event_ids"]


def test_atomic_event_audit_exposes_multi_row_and_anchor_outside_risks():
    diagnostic = _load_script_module()
    events = [
        {
            "event_id": 0,
            "candidate_id": "target",
            "cross_ids": [4],
            "question_bbox": [0.1, 0.1, 0.5, 0.5],
            "confidence": 0.9,
        }
    ]
    anchors = [{"cross_id": 4, "bbox": [0.7, 0.2, 0.75, 0.25]}]
    ocr_lines = [
        {"text": "row one", "bbox": [0.15, 0.12, 0.4, 0.16], "confidence": 0.9},
        {"text": "row two", "bbox": [0.15, 0.24, 0.4, 0.28], "confidence": 0.9},
        {"text": "row three", "bbox": [0.15, 0.4, 0.4, 0.44], "confidence": 0.9},
    ]

    refined, audit = diagnostic.build_atomic_question_events(
        events=events,
        ocr_lines=ocr_lines,
        config=_config(),
        anchors=anchors,
    )

    assert refined[0]["question_bbox"] == [0.1, 0.1, 0.5, 0.5]
    assert audit["multi_ocr_row_event_ids"] == [0]
    assert audit["anchor_outside_event_ids"] == [0]
    assert audit["event_ocr_row_audits"][0]["ocr_row_group_count"] == 3


def test_atomic_events_do_not_partition_away_a_previously_contained_anchor():
    diagnostic = _load_script_module()
    events = [
        {
            "event_id": 0,
            "candidate_id": "left",
            "cross_ids": [0],
            "question_bbox": [0.1, 0.2, 0.5, 0.4],
            "confidence": 0.9,
        },
        {
            "event_id": 1,
            "candidate_id": "right",
            "cross_ids": [1],
            "question_bbox": [0.3, 0.2, 0.7, 0.4],
            "confidence": 0.9,
        },
    ]
    anchors = [
        {"cross_id": 0, "bbox": [0.43, 0.27, 0.47, 0.31]},
        {"cross_id": 1, "bbox": [0.55, 0.27, 0.59, 0.31]},
    ]

    refined, audit = diagnostic.build_atomic_question_events(
        events=events,
        ocr_lines=[],
        config=_config(),
        anchors=anchors,
    )

    assert refined[0]["question_bbox"] == [0.1, 0.2, 0.45, 0.4]
    assert refined[1]["question_bbox"][0] == 0.388
    assert audit["partitioned_event_ids"] == [0, 1]
    assert audit["anchor_outside_event_ids"] == []
