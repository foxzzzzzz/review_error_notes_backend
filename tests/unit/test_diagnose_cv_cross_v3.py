import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "diagnose_cv_cross_v3.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("diagnose_cv_cross_v3", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(candidate_id, bbox, center):
    return {
        "candidate_id": candidate_id,
        "bbox": bbox,
        "center": center,
        "min_arm_density": 0.5,
        "arm_densities": [0.5, 0.5, 0.5, 0.5],
        "center_density": 0.8,
    }


def _config():
    return {
        "background_percentile": 90,
        "normalized_red_intersection_threshold": 0.03,
        "intersection_candidate_limit": 12,
        "intersection_min_spacing_ratio": 0.005,
        "continuity_inner_radius_ratio": 0.05,
        "continuity_outer_radius_ratio": 0.25,
        "continuity_band_ratio": 0.02,
        "continuity_angle_min_degrees": 20,
        "continuity_angle_max_degrees": 70,
        "continuity_angle_step_degrees": 5,
        "continuity_score_weight": 0.5,
        "minimum_candidate_quality_score": 0.3,
    }


def test_v3_keeps_strong_red_candidate_even_when_it_touches_page_edge():
    diagnostic = _load_script_module()
    pixels = np.full((101, 101, 3), 220, dtype=np.uint8)
    pixels[35:65, 0:18] = [210, 80, 80]
    candidate = _candidate(0, [0.0, 0.3, 0.2, 0.7], [0.09, 0.5])

    result = diagnostic.filter_cross_candidates_v3(
        pixels, [candidate], _config()
    )

    assert [item["candidate_id"] for item in result["candidates"]] == [0]


def test_v3_keeps_strong_red_candidate_with_low_diagonal_continuity():
    diagnostic = _load_script_module()
    pixels = np.full((101, 101, 3), 220, dtype=np.uint8)
    pixels[42:59, 42:59] = [210, 80, 80]
    candidate = _candidate(0, [0.35, 0.35, 0.65, 0.65], [0.5, 0.5])

    result = diagnostic.filter_cross_candidates_v3(
        pixels, [candidate], _config()
    )

    assert [item["candidate_id"] for item in result["candidates"]] == [0]


def test_v3_rejects_candidate_when_color_and_cross_geometry_are_both_weak():
    diagnostic = _load_script_module()
    pixels = np.full((101, 101, 3), [220, 218, 210], dtype=np.uint8)
    pixels[42:59, 42:59] = [205, 190, 180]
    candidate = _candidate(0, [0.35, 0.35, 0.65, 0.65], [0.5, 0.5])

    result = diagnostic.filter_cross_candidates_v3(
        pixels, [candidate], _config()
    )

    assert result["candidates"] == []
    assert result["rejected_candidates"][0]["reasons"] == [
        "insufficient_combined_cross_evidence"
    ]


def test_v3_recenters_geometry_search_on_cross_inside_candidate_box():
    diagnostic = _load_script_module()
    image = Image.new("RGB", (101, 101), (220, 220, 220))
    draw = ImageDraw.Draw(image)
    draw.line((55, 55, 85, 85), fill=(210, 40, 40), width=3)
    draw.line((55, 85, 85, 55), fill=(210, 40, 40), width=3)
    draw.point((70, 70), fill=(255, 20, 20))
    candidate = _candidate(0, [0.4, 0.4, 0.9, 0.9], [0.45, 0.45])

    result = diagnostic.filter_cross_candidates_v3(
        np.asarray(image), [candidate], _config()
    )

    audit = result["candidate_audits"][0]
    assert result["candidates"]
    assert audit["best_opposite_arm_continuity"] >= 0.8
    assert abs(audit["best_intersection_center"][0] - 0.7) <= 0.05
    assert abs(audit["best_intersection_center"][1] - 0.7) <= 0.05
