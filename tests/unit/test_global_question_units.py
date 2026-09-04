import importlib.util
import inspect
import json
from pathlib import Path

import cv2
import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = BACKEND_ROOT / "scripts" / "global_question_units.py"
CONFIG_PATH = BACKEND_ROOT / "scripts" / "global_question_unit_config.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("global_question_units", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _ocr(line_id, bbox, text):
    return {
        "ocr_line_id": line_id,
        "bbox": bbox,
        "text": text,
        "confidence": 0.95,
    }


def _white_page_with_red_grid(*, rows, columns):
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    red = (0, 0, 180)
    for y in rows:
        cv2.line(image, (round(columns[0] * 1000), round(y * 1000)),
                 (round(columns[-1] * 1000), round(y * 1000)), red, 5)
    for x in columns:
        cv2.line(image, (round(x * 1000), round(rows[0] * 1000)),
                 (round(x * 1000), round(rows[-1] * 1000)), red, 5)
    return image


def test_stacked_siblings_have_distinct_stable_ids():
    units = _load_module()
    image = _white_page_with_red_grid(
        rows=[0.10, 0.30, 0.50], columns=[0.10, 0.90]
    )
    lines = [
        _ocr(0, [0.15, 0.14, 0.55, 0.20], "first"),
        _ocr(1, [0.15, 0.34, 0.55, 0.40], "second"),
    ]

    result = units.build_global_question_units(image, lines, _config())

    assert [item["question_unit_id"] for item in result["units"]] == [
        "U-S01-R01-C01",
        "U-S01-R02-C01",
    ]
    assert result["units"][0]["unit_bbox"][3] <= result["units"][1]["unit_bbox"][1]


def test_side_by_side_siblings_have_distinct_column_ids():
    units = _load_module()
    image = _white_page_with_red_grid(
        rows=[0.10, 0.30], columns=[0.10, 0.50, 0.90]
    )
    lines = [
        _ocr(0, [0.14, 0.14, 0.42, 0.20], "left"),
        _ocr(1, [0.58, 0.14, 0.86, 0.20], "right"),
    ]

    result = units.build_global_question_units(image, lines, _config())

    assert [item["question_unit_id"] for item in result["units"]] == [
        "U-S01-R01-C01",
        "U-S01-R01-C02",
    ]


def test_repeated_build_is_byte_stable():
    units = _load_module()
    image = _white_page_with_red_grid(
        rows=[0.10, 0.30, 0.50], columns=[0.10, 0.50, 0.90]
    )
    lines = [
        _ocr(5, [0.58, 0.34, 0.86, 0.40], "fourth"),
        _ocr(2, [0.14, 0.14, 0.42, 0.20], "first"),
    ]

    first = units.build_global_question_units(image, lines, _config())
    second = units.build_global_question_units(image, list(reversed(lines)), _config())

    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )


def test_grid_only_units_survive_missing_ocr_with_risk_flag():
    units = _load_module()
    image = _white_page_with_red_grid(
        rows=[0.10, 0.30], columns=[0.10, 0.50, 0.90]
    )

    result = units.build_global_question_units(image, [], _config())

    assert result["units"]
    assert all("ocr_missing" in item["risk_flags"] for item in result["units"])


def test_answer_grid_component_expands_to_whole_question_context():
    units = _load_module()
    image = _white_page_with_red_grid(
        rows=[0.20, 0.27], columns=[0.10, 0.35]
    )
    lines = [_ocr(0, [0.13, 0.21, 0.32, 0.26], "answer")]

    result = units.build_global_question_units(image, lines, _config())

    assert len(result["units"]) == 1
    assert result["units"][0]["unit_bbox"][1] < 0.18
    assert result["units"][0]["unit_bbox"][3] > 0.30


def test_ocr_seed_creates_local_unit_when_red_grid_is_broken():
    units = _load_module()
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    lines = [_ocr(0, [0.10, 0.40, 0.30, 0.45], "dong gua")]

    result = units.build_global_question_units(image, lines, _config())

    assert len(result["units"]) == 1
    bbox = result["units"][0]["unit_bbox"]
    assert 0.0 < bbox[0] < 0.10
    assert 0.30 < bbox[2] < 0.50
    assert bbox[1] < 0.40 < bbox[3]


def test_neighboring_layout_components_do_not_overlap():
    units = _load_module()
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    red = (0, 0, 180)
    cv2.rectangle(image, (100, 200), (300, 270), red, 5)
    cv2.rectangle(image, (350, 200), (550, 270), red, 5)
    lines = [
        _ocr(0, [0.12, 0.21, 0.28, 0.26], "left"),
        _ocr(1, [0.37, 0.21, 0.53, 0.26], "right"),
    ]

    result = units.build_global_question_units(image, lines, _config())

    assert len(result["units"]) == 2
    assert result["units"][0]["unit_bbox"][2] <= result["units"][1]["unit_bbox"][0]


def _unit(unit_id, bbox):
    return {
        "question_unit_id": unit_id,
        "unit_bbox": bbox,
        "context_bbox": bbox,
        "ocr_line_ids": [],
        "layout_evidence": "grid",
        "risk_flags": [],
    }


def _anchor(cross_id, center_x, center_y):
    return {
        "cross_id": cross_id,
        "bbox": [center_x - 0.01, center_y - 0.01, center_x + 0.01, center_y + 0.01],
    }


def test_anchor_mapping_keeps_at_most_three_ranked_units():
    units = _load_module()
    page_units = [
        _unit("U-S01-R01-C01", [0.0, 0.0, 0.5, 0.5]),
        _unit("U-S01-R01-C02", [0.5, 0.0, 1.0, 0.5]),
        _unit("U-S01-R02-C01", [0.0, 0.5, 0.5, 1.0]),
        _unit("U-S01-R02-C02", [0.5, 0.5, 1.0, 1.0]),
    ]

    mapped = units.map_anchors_to_units(
        page_units, [_anchor(7, 0.49, 0.25)], _config()
    )

    candidates = mapped["anchor_candidates"]["7"]
    assert 1 <= len(candidates) <= 3
    assert candidates == sorted(
        candidates,
        key=lambda item: (-item["score"], item["question_unit_id"]),
    )
    assert candidates[0]["question_unit_id"] == "U-S01-R01-C01"


def test_unmapped_anchor_is_audited_not_dropped():
    units = _load_module()
    page_units = [_unit("U-S01-R01-C01", [0.0, 0.0, 0.2, 0.2])]

    mapped = units.map_anchors_to_units(
        page_units, [_anchor(9, 0.95, 0.95)], _config()
    )

    assert mapped["anchor_candidates"]["9"] == []
    assert mapped["unassigned_anchors"] == [
        {"cross_id": 9, "reason": "no_unit_within_distance"}
    ]


def test_truth_is_not_an_input_to_runtime_geometry():
    units = _load_module()

    unit_signature = inspect.signature(units.build_global_question_units)
    anchor_signature = inspect.signature(units.map_anchors_to_units)

    assert "truth_regions" not in unit_signature.parameters
    assert "truth_regions" not in anchor_signature.parameters


def test_candidate_oracle_reports_identity_recall_without_mutating_units():
    units = _load_module()
    page_units = [
        _unit("U-S01-R01-C01", [0.1, 0.1, 0.4, 0.4]),
        _unit("U-S01-R01-C02", [0.6, 0.1, 0.9, 0.4]),
    ]
    mapping = {
        "anchor_candidates": {
            "1": [{"question_unit_id": "U-S01-R01-C01"}],
        },
        "unassigned_anchors": [],
    }
    truth = [
        {"truth_id": "T1", "source_bbox_normalized": [0.12, 0.12, 0.38, 0.38]},
        {"truth_id": "T2", "source_bbox_normalized": [0.62, 0.12, 0.88, 0.38]},
    ]
    original = json.dumps(page_units, sort_keys=True)

    audit = units.compare_unit_candidates_to_truth(
        page_units, mapping, truth, _config()
    )

    assert audit["matched_truth_ids"] == ["T1"]
    assert audit["missed_truth_ids"] == ["T2"]
    assert audit["truth_recall"] == 0.5
    assert json.dumps(page_units, sort_keys=True) == original


def test_atomic_oracle_rejects_unit_that_covers_two_truth_regions():
    units = _load_module()
    page_units = [
        _unit("ATOMIC", [0.10, 0.10, 0.40, 0.40]),
        _unit("WIDE", [0.10, 0.10, 0.90, 0.40]),
    ]
    truth = [
        {"truth_id": "T1", "source_bbox_normalized": [0.12, 0.12, 0.38, 0.38]},
        {"truth_id": "T2", "source_bbox_normalized": [0.62, 0.12, 0.88, 0.38]},
    ]

    audit = units.audit_fixed_units(page_units, truth, _config())

    assert audit["matched_truth_ids"] == ["T1", "T2"]
    assert audit["truth_recall"] == 1.0
    assert audit["atomic_matched_truth_ids"] == ["T1"]
    assert audit["atomic_missed_truth_ids"] == ["T2"]
    assert audit["atomic_truth_recall"] == 0.5
    assert audit["non_atomic_unit_ids"] == ["WIDE"]
