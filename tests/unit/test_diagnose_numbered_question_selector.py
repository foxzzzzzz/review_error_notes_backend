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
