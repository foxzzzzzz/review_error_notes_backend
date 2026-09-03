import importlib.util
import json
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = BACKEND_ROOT / "scripts" / "global_question_unit_convergence.py"
CONFIG_PATH = BACKEND_ROOT / "scripts" / "global_question_unit_config.json"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "global_question_unit_convergence", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _event(
    unit_id,
    bbox,
    *,
    anchors=(1,),
    ocr_tokens=("same",),
    round_index=1,
    boundary_fits=("complete",),
):
    return {
        "question_unit_id": unit_id,
        "unit_bbox": bbox,
        "anchor_ids": list(anchors),
        "ocr_tokens": list(ocr_tokens),
        "source_round": round_index,
        "boundary_fits": list(boundary_fits),
    }


def test_retry_union_merges_exact_stable_ids_and_preserves_round_evidence():
    convergence = _load_module()

    result = convergence.consolidate_retry_events(
        [
            [_event("U1", [0.1, 0.1, 0.4, 0.3], round_index=1)],
            [_event("U1", [0.1, 0.1, 0.4, 0.3], round_index=2)],
        ],
        _config(),
    )

    assert len(result["events"]) == 1
    assert len(result["stable_id_union_events"]) == 1
    assert result["events"][0]["source_rounds"] == [1, 2]
    assert result["events"][0]["source_unit_ids"] == ["U1"]
    assert result["audit"]["exact_id_merge_count"] == 1


def test_ocr_tokens_ignore_spacing_and_punctuation_but_keep_content():
    convergence = _load_module()
    unit = {"ocr_line_ids": [4]}

    first = convergence.build_unit_ocr_tokens(
        unit, [{"ocr_line_id": 4, "text": "第 3 题：苹果"}]
    )
    second = convergence.build_unit_ocr_tokens(
        unit, [{"ocr_line_id": 4, "text": "第3题,苹果"}]
    )

    assert first == second
    assert first


def test_retry_union_merges_only_when_ocr_space_and_anchor_evidence_agree():
    convergence = _load_module()

    result = convergence.consolidate_retry_events(
        [
            [_event("U-wide", [0.1, 0.1, 0.5, 0.4], round_index=1)],
            [_event("U-tight", [0.12, 0.11, 0.48, 0.38], round_index=2)],
        ],
        _config(),
    )

    assert len(result["events"]) == 1
    assert len(result["stable_id_union_events"]) == 2
    assert result["events"][0]["question_unit_id"] == "U-tight"
    assert result["events"][0]["source_unit_ids"] == ["U-tight", "U-wide"]
    assert result["audit"]["conservative_ocr_geometry_merge_count"] == 1


def test_retry_union_never_merges_spatially_separate_ocr_twins():
    convergence = _load_module()

    result = convergence.consolidate_retry_events(
        [
            [_event("left", [0.05, 0.1, 0.35, 0.3], anchors=(1,))],
            [_event("right", [0.6, 0.1, 0.9, 0.3], anchors=(2,), round_index=2)],
        ],
        _config(),
    )

    assert [item["question_unit_id"] for item in result["events"]] == [
        "left",
        "right",
    ]
    assert result["audit"]["conservative_ocr_geometry_merge_count"] == 0


def test_retry_union_never_merges_overlapping_units_with_different_ocr():
    convergence = _load_module()

    result = convergence.consolidate_retry_events(
        [
            [_event("first", [0.1, 0.1, 0.5, 0.4], ocr_tokens=("first",))],
            [
                _event(
                    "sibling",
                    [0.12, 0.11, 0.48, 0.38],
                    ocr_tokens=("second",),
                    round_index=2,
                )
            ],
        ],
        _config(),
    )

    assert len(result["events"]) == 2


def test_boundary_metadata_auto_uses_matching_narrow_unit_and_flags_sibling_option():
    convergence = _load_module()
    units = {
        "wide": {
            "question_unit_id": "wide",
            "unit_bbox": [0.1, 0.1, 0.6, 0.4],
            "ocr_tokens": ["question-a"],
        },
        "tight": {
            "question_unit_id": "tight",
            "unit_bbox": [0.12, 0.12, 0.46, 0.36],
            "ocr_tokens": ["question-a"],
        },
        "sibling": {
            "question_unit_id": "sibling",
            "unit_bbox": [0.3, 0.12, 0.58, 0.36],
            "ocr_tokens": ["question-b"],
        },
    }
    events = [_event("wide", [0.1, 0.1, 0.6, 0.4])]
    candidates = {
        "1": [
            {"question_unit_id": "wide", "rank": 1, "anchor_in_unit": True},
            {"question_unit_id": "tight", "rank": 2, "anchor_in_unit": True},
            {"question_unit_id": "sibling", "rank": 3, "anchor_in_unit": True},
        ]
    }

    result = convergence.build_boundary_metadata(
        events, units, candidates, _config()
    )

    assert result[0]["display_unit_id"] == "tight"
    assert result[0]["display_bbox"] == [0.12, 0.12, 0.46, 0.36]
    assert result[0]["boundary_status"] == "boundary_ambiguous"
    assert result[0]["interaction_option_unit_ids"] == ["tight", "wide", "sibling"]


def test_boundary_metadata_does_not_offer_spatially_separate_sibling():
    convergence = _load_module()
    units = {
        "chosen": {
            "question_unit_id": "chosen",
            "unit_bbox": [0.05, 0.1, 0.35, 0.3],
            "ocr_tokens": ["same"],
        },
        "far": {
            "question_unit_id": "far",
            "unit_bbox": [0.6, 0.1, 0.9, 0.3],
            "ocr_tokens": ["same"],
        },
    }
    events = [_event("chosen", [0.05, 0.1, 0.35, 0.3])]
    candidates = {
        "1": [
            {"question_unit_id": "chosen", "rank": 1, "anchor_in_unit": True},
            {"question_unit_id": "far", "rank": 2, "anchor_in_unit": True},
        ]
    }

    result = convergence.build_boundary_metadata(
        events, units, candidates, _config()
    )

    assert result[0]["boundary_status"] == "automatic"
    assert result[0]["interaction_option_unit_ids"] == ["chosen"]


def test_boundary_metadata_ignores_partially_overlapping_neighbor():
    convergence = _load_module()
    units = {
        "chosen": {
            "question_unit_id": "chosen",
            "unit_bbox": [0.1, 0.1, 0.5, 0.4],
            "ocr_tokens": ["first"],
        },
        "neighbor": {
            "question_unit_id": "neighbor",
            "unit_bbox": [0.36, 0.1, 0.76, 0.4],
            "ocr_tokens": ["second"],
        },
    }
    events = [_event("chosen", [0.1, 0.1, 0.5, 0.4])]
    candidates = {
        "1": [
            {"question_unit_id": "chosen", "rank": 1, "anchor_in_unit": True},
            {"question_unit_id": "neighbor", "rank": 2, "anchor_in_unit": True},
        ]
    }

    result = convergence.build_boundary_metadata(
        events, units, candidates, _config()
    )

    assert result[0]["boundary_status"] == "automatic"
    assert result[0]["interaction_option_unit_ids"] == ["chosen"]


def test_semantic_uncertainty_alone_is_not_presented_as_boundary_intrusion():
    convergence = _load_module()
    units = {
        "chosen": {
            "question_unit_id": "chosen",
            "unit_bbox": [0.1, 0.1, 0.5, 0.4],
            "ocr_tokens": ["question"],
        }
    }
    events = [
        _event(
            "chosen",
            [0.1, 0.1, 0.5, 0.4],
            boundary_fits=("uncertain",),
        )
    ]

    result = convergence.build_boundary_metadata(
        events,
        units,
        {"1": [{"question_unit_id": "chosen", "rank": 1, "anchor_in_unit": True}]},
        _config(),
    )

    assert result[0]["boundary_status"] == "automatic"


def test_boundary_metadata_flags_multiple_anchors_with_different_rank_one_units():
    convergence = _load_module()
    units = {
        "combined": {
            "question_unit_id": "combined",
            "unit_bbox": [0.1, 0.1, 0.8, 0.4],
            "ocr_tokens": ["both"],
        },
        "left": {
            "question_unit_id": "left",
            "unit_bbox": [0.1, 0.1, 0.42, 0.4],
            "ocr_tokens": ["left"],
        },
        "right": {
            "question_unit_id": "right",
            "unit_bbox": [0.48, 0.1, 0.8, 0.4],
            "ocr_tokens": ["right"],
        },
    }
    events = [_event("combined", [0.1, 0.1, 0.8, 0.4], anchors=(1, 2))]
    candidates = {
        "1": [
            {"question_unit_id": "left", "rank": 1, "anchor_in_unit": True},
            {"question_unit_id": "combined", "rank": 2, "anchor_in_unit": True},
        ],
        "2": [
            {"question_unit_id": "right", "rank": 1, "anchor_in_unit": True},
            {"question_unit_id": "combined", "rank": 2, "anchor_in_unit": True},
        ],
    }

    result = convergence.build_boundary_metadata(
        events, units, candidates, _config()
    )

    assert result[0]["boundary_status"] == "boundary_ambiguous"
    assert "multi_anchor_rank_one_conflict" in result[0]["boundary_reasons"]
    assert set(result[0]["interaction_option_unit_ids"]) == {
        "combined",
        "left",
        "right",
    }


def test_boundary_metadata_flags_selected_unit_that_excludes_anchor_center():
    convergence = _load_module()
    units = {
        "selected": {
            "question_unit_id": "selected",
            "unit_bbox": [0.1, 0.1, 0.4, 0.3],
            "ocr_tokens": ["question"],
        },
        "near": {
            "question_unit_id": "near",
            "unit_bbox": [0.4, 0.1, 0.7, 0.3],
            "ocr_tokens": ["neighbor"],
        },
    }
    events = [_event("selected", [0.1, 0.1, 0.4, 0.3])]
    candidates = {
        "1": [
            {"question_unit_id": "selected", "rank": 1, "anchor_in_unit": False},
            {"question_unit_id": "near", "rank": 2, "anchor_in_unit": False},
        ]
    }

    result = convergence.build_boundary_metadata(
        events, units, candidates, _config()
    )

    assert result[0]["boundary_status"] == "boundary_ambiguous"
    assert result[0]["boundary_reasons"] == ["selected_unit_excludes_anchor"]
    assert result[0]["interaction_option_unit_ids"] == ["selected", "near"]


def test_boundary_metadata_flags_non_primary_anchor_containing_unit():
    convergence = _load_module()
    units = {
        "primary": {
            "question_unit_id": "primary",
            "unit_bbox": [0.1, 0.1, 0.4, 0.3],
            "ocr_tokens": ["first"],
        },
        "selected": {
            "question_unit_id": "selected",
            "unit_bbox": [0.2, 0.1, 0.55, 0.3],
            "ocr_tokens": ["second"],
        },
    }
    events = [_event("selected", [0.2, 0.1, 0.55, 0.3])]
    candidates = {
        "1": [
            {"question_unit_id": "primary", "rank": 1, "anchor_in_unit": True},
            {"question_unit_id": "selected", "rank": 2, "anchor_in_unit": True},
        ]
    }

    result = convergence.build_boundary_metadata(
        events, units, candidates, _config()
    )

    assert result[0]["boundary_status"] == "boundary_ambiguous"
    assert result[0]["boundary_reasons"] == ["non_primary_anchor_unit"]
    assert result[0]["interaction_option_unit_ids"] == ["selected", "primary"]


def test_boundary_metadata_flags_overlapping_windows_above_and_below_anchor_unit():
    convergence = _load_module()
    units = {
        "middle": {
            "question_unit_id": "middle",
            "unit_bbox": [0.3, 0.2, 0.6, 0.4],
            "ocr_tokens": ["middle"],
        },
        "above": {
            "question_unit_id": "above",
            "unit_bbox": [0.3, 0.1, 0.6, 0.3],
            "ocr_tokens": ["above"],
        },
        "below": {
            "question_unit_id": "below",
            "unit_bbox": [0.3, 0.3, 0.6, 0.5],
            "ocr_tokens": ["below"],
        },
    }
    events = [_event("middle", [0.3, 0.2, 0.6, 0.4])]
    candidates = {
        "1": [
            {"question_unit_id": "middle", "rank": 1, "anchor_in_unit": True},
            {"question_unit_id": "below", "rank": 2, "anchor_in_unit": True},
            {"question_unit_id": "above", "rank": 3, "anchor_in_unit": True},
        ]
    }

    result = convergence.build_boundary_metadata(
        events, units, candidates, _config()
    )

    assert result[0]["boundary_status"] == "boundary_ambiguous"
    assert "straddled_anchor_window" in result[0]["boundary_reasons"]
    assert result[0]["interaction_option_unit_ids"] == ["middle", "below", "above"]
