import pytest
from PIL import Image, ImageDraw

from app.services.vision_recognition import ErrorMark


def _mark(bbox=None, confidence=0.95):
    return ErrorMark(
        mark_id=0,
        mark_type="circle",
        bbox=bbox or [0.15, 0.15, 0.45, 0.45],
        confidence=confidence,
    )


def test_groups_neighboring_red_fragments_without_merging_distant_regions():
    from app.services.error_mark_validation import (
        RedMarkRegion,
        group_red_evidence_regions,
    )

    regions = [
        RedMarkRegion(bbox=[0.10, 0.10, 0.12, 0.12], pixel_count=20, area_ratio=0.0004, thinness_ratio=1),
        RedMarkRegion(bbox=[0.13, 0.10, 0.15, 0.12], pixel_count=30, area_ratio=0.0004, thinness_ratio=1),
        RedMarkRegion(bbox=[0.70, 0.70, 0.72, 0.72], pixel_count=40, area_ratio=0.0004, thinness_ratio=1),
    ]

    grouped, diagnostic = group_red_evidence_regions(
        regions, max_gap_ratio=0.03, max_group_area_ratio=0.08
    )

    assert len(grouped) == 2
    assert grouped[0].bbox == [0.10, 0.10, 0.15, 0.12]
    assert grouped[0].pixel_count == 50
    assert diagnostic == {"raw_component_count": 3, "evidence_group_count": 2}


def test_merges_mark_attempts_without_losing_different_shape_types():
    from app.services.error_mark_validation import merge_error_mark_attempts

    cross = ErrorMark(mark_id=0, mark_type="cross", bbox=[0.4, 0.2, 0.46, 0.28], cross_bbox=[0.4, 0.2, 0.46, 0.28], confidence=0.95)
    circle = ErrorMark(mark_id=0, mark_type="circle", bbox=[0.3, 0.25, 0.5, 0.42], circle_bbox=[0.3, 0.25, 0.5, 0.42], confidence=0.94)

    merged, diagnostic = merge_error_mark_attempts(
        [[cross], [circle]], dedup_iou_threshold=0.8
    )

    assert [mark.mark_type for mark in merged] == ["cross", "circle"]
    assert [mark.mark_id for mark in merged] == [0, 1]
    assert diagnostic["attempt_primitive_counts"] == [1, 1]
    assert diagnostic["cross_attempt_deduplicated_count"] == 0


def test_merges_duplicate_same_shape_across_attempts():
    from app.services.error_mark_validation import merge_error_mark_attempts

    first = _mark([0.2, 0.2, 0.4, 0.4])
    second = _mark([0.21, 0.2, 0.4, 0.4]).model_copy(update={"confidence": 0.96})

    merged, diagnostic = merge_error_mark_attempts(
        [[first], [second]], dedup_iou_threshold=0.8
    )

    assert len(merged) == 1
    assert merged[0].confidence == 0.96
    assert diagnostic["cross_attempt_deduplicated_count"] == 1


def test_accepts_mark_box_with_red_pixels(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "marks.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 40, 40), outline=(220, 30, 30), width=5)
    image.save(image_path)

    valid, rejected, _diagnostics = filter_valid_error_marks(
        str(image_path),
        [_mark()],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.05,
    )

    assert [mark.mark_id for mark in valid] == [0]
    assert rejected == []


def test_reports_pixel_evidence_and_rejection_reason_for_each_mark(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "diagnostics.png"
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 40, 40), fill=(220, 30, 30))
    image.save(image_path)

    valid, rejected, diagnostics = filter_valid_error_marks(
        str(image_path),
        [
            _mark([0.2, 0.2, 0.4, 0.4]),
            _mark([0.6, 0.6, 0.8, 0.8]).model_copy(update={"mark_id": 1}),
            _mark([0.2, 0.2, 0.4, 0.4], confidence=0.5).model_copy(
                update={"mark_id": 2}
            ),
        ],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.0,
    )

    assert [mark.mark_id for mark in valid] == [0]
    assert rejected == [1, 2]
    assert diagnostics[0] == {
        "mark_id": 0,
        "confidence": 0.95,
        "confidence_threshold": 0.85,
        "pixel_box": [20, 20, 40, 40],
        "red_pixel_count": 400,
        "pixel_count": 400,
        "red_pixel_ratio": 1.0,
        "red_pixel_min_ratio": 0.01,
        "accepted": True,
        "reason": "accepted",
        "mark_type": "circle",
    }
    assert diagnostics[1]["red_pixel_ratio"] == 0.0
    assert diagnostics[1]["reason"] == "insufficient_red_pixels"
    assert diagnostics[2]["red_pixel_ratio"] == 1.0
    assert diagnostics[2]["reason"] == "low_confidence"


def test_rejects_white_region_and_low_confidence_mark(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "white.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    valid, rejected, _diagnostics = filter_valid_error_marks(
        str(image_path),
        [_mark(), _mark(confidence=0.5).model_copy(update={"mark_id": 1})],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.05,
    )

    assert valid == []
    assert rejected == [0, 1]


def test_cross_circle_requires_red_pixels_in_both_components(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "cross-circle.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((55, 10, 65, 20), fill=(220, 20, 20), width=3)
    draw.line((65, 10, 55, 20), fill=(220, 20, 20), width=3)
    draw.ellipse((20, 25, 70, 70), outline=(220, 20, 20), width=4)
    image.save(image_path)
    mark = ErrorMark(
        mark_id=0,
        mark_type="cross_circle",
        bbox=[0.2, 0.1, 0.7, 0.7],
        cross_bbox=[0.54, 0.09, 0.66, 0.21],
        circle_bbox=[0.19, 0.24, 0.71, 0.71],
        confidence=0.95,
    )

    valid, rejected, diagnostics = filter_valid_error_marks(
        str(image_path),
        [mark],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.005,
        expansion_ratio=0.02,
    )

    assert [item.mark_id for item in valid] == [0]
    assert rejected == []
    assert diagnostics[0]["component_validation"]["cross"]["accepted"] is True
    assert diagnostics[0]["component_validation"]["circle"]["accepted"] is True


def test_cross_circle_is_rejected_when_circle_component_has_no_red_pixels(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "cross-without-circle.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((55, 10, 65, 20), fill=(220, 20, 20), width=3)
    draw.line((65, 10, 55, 20), fill=(220, 20, 20), width=3)
    image.save(image_path)
    mark = ErrorMark(
        mark_id=0,
        mark_type="cross_circle",
        bbox=[0.2, 0.1, 0.7, 0.7],
        cross_bbox=[0.54, 0.09, 0.66, 0.21],
        circle_bbox=[0.19, 0.24, 0.71, 0.71],
        confidence=0.95,
    )

    valid, rejected, diagnostics = filter_valid_error_marks(
        str(image_path),
        [mark],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.005,
        expansion_ratio=0.02,
    )

    assert valid == []
    assert rejected == [0]
    assert diagnostics[0]["reason"] == "invalid_component_pixels"
    assert diagnostics[0]["component_validation"]["cross"]["accepted"] is True
    assert diagnostics[0]["component_validation"]["circle"]["accepted"] is False


def test_cross_circle_degrades_to_valid_cross_when_circle_has_no_red_pixels(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks
    from app.services.vision_recognition import ErrorMark

    image_path = tmp_path / "partial-cross-circle.png"
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line((30, 20, 70, 60), fill=(220, 20, 20), width=6)
    draw.line((70, 20, 30, 60), fill=(220, 20, 20), width=6)
    image.save(image_path)
    mark = ErrorMark(
        mark_id=0,
        mark_type="cross_circle",
        bbox=[0.1, 0.1, 0.9, 0.8],
        cross_bbox=[0.1, 0.1, 0.4, 0.6],
        circle_bbox=[0.6, 0.1, 0.9, 0.6],
        confidence=0.95,
    )

    valid, rejected, diagnostics = filter_valid_error_marks(
        str(image_path),
        [mark],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.0,
        component_fallback_enabled=True,
    )

    assert rejected == []
    assert [candidate.mark_type for candidate in valid] == ["cross"]
    assert valid[0].bbox == mark.cross_bbox
    assert diagnostics[0]["fallback_type"] == "cross"


def test_cross_circle_splits_components_when_union_does_not_contain_them(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks
    from app.services.vision_recognition import ErrorMark

    image_path = tmp_path / "misgrouped-cross-circle.png"
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 50, 50), fill=(220, 20, 20))
    draw.rectangle((140, 20, 170, 50), fill=(220, 20, 20))
    image.save(image_path)
    mark = ErrorMark(
        mark_id=0,
        mark_type="cross_circle",
        bbox=[0.05, 0.1, 0.35, 0.5],
        cross_bbox=[0.1, 0.15, 0.25, 0.45],
        circle_bbox=[0.7, 0.15, 0.85, 0.45],
        confidence=0.95,
    )

    valid, rejected, diagnostics = filter_valid_error_marks(
        str(image_path),
        [mark],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.0,
        component_fallback_enabled=True,
    )

    assert rejected == []
    assert [candidate.mark_type for candidate in valid] == ["cross", "circle"]
    assert diagnostics[0]["components_within_union"] is False
    assert diagnostics[0]["fallback_type"] == "split_components"
    assert diagnostics[0]["component_validation"]["cross"]["accepted"] is True
    assert diagnostics[0]["component_validation"]["circle"]["accepted"] is True


def test_cross_circle_splits_far_components_even_when_union_contains_both(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks
    from app.services.vision_recognition import ErrorMark

    image_path = tmp_path / "far-cross-circle.png"
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 50, 50), fill=(220, 20, 20))
    draw.rectangle((140, 20, 170, 50), fill=(220, 20, 20))
    image.save(image_path)
    mark = ErrorMark(
        mark_id=0,
        mark_type="cross_circle",
        bbox=[0.05, 0.1, 0.9, 0.5],
        cross_bbox=[0.1, 0.15, 0.25, 0.45],
        circle_bbox=[0.7, 0.15, 0.85, 0.45],
        confidence=0.95,
    )

    valid, rejected, diagnostics = filter_valid_error_marks(
        str(image_path),
        [mark],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.0,
        component_fallback_enabled=True,
        component_pair_max_distance_ratio=0.04,
    )

    assert rejected == []
    assert [candidate.mark_type for candidate in valid] == ["cross", "circle"]
    assert diagnostics[0]["components_within_union"] is True
    assert diagnostics[0]["components_pairable"] is False
    assert diagnostics[0]["fallback_type"] == "split_components"


def test_normalize_error_mark_groups_deduplicates_overlapping_marks():
    from app.services.error_mark_validation import normalize_error_mark_groups

    marks = [
        _mark([0.2, 0.2, 0.5, 0.5]),
        _mark([0.2, 0.2, 0.5, 0.5]).model_copy(update={"mark_id": 1}),
        _mark([0.28, 0.28, 0.34, 0.34]).model_copy(update={"mark_id": 2}),
    ]

    groups, diagnostic = normalize_error_mark_groups(
        marks,
        dedup_iou_threshold=0.8,
        pair_max_distance_ratio=0.12,
    )

    assert [group.mark_id for group in groups] == [0]
    assert diagnostic == {
        "raw_mark_count": 3,
        "correction_group_count": 1,
        "paired_group_count": 0,
        "single_mark_group_count": 1,
        "deduplicated_mark_count": 2,
        "review_required_mark_ids": [],
    }


def test_normalize_error_mark_groups_does_not_force_ambiguous_pair():
    from app.services.error_mark_validation import normalize_error_mark_groups

    marks = [
        ErrorMark(
            mark_id=0,
            mark_type="cross",
            bbox=[0.2, 0.2, 0.25, 0.25],
            confidence=0.95,
        ),
        ErrorMark(
            mark_id=1,
            mark_type="cross",
            bbox=[0.35, 0.2, 0.4, 0.25],
            confidence=0.95,
        ),
        ErrorMark(
            mark_id=2,
            mark_type="circle",
            bbox=[0.27, 0.2, 0.33, 0.25],
            confidence=0.95,
        ),
    ]

    groups, diagnostic = normalize_error_mark_groups(
        marks,
        dedup_iou_threshold=0.8,
        pair_max_distance_ratio=0.12,
    )

    assert [group.mark_type for group in groups] == ["cross", "cross", "circle"]
    assert [group.mark_id for group in groups] == [0, 1, 2]
    assert diagnostic["paired_group_count"] == 0


def test_normalize_error_mark_groups_pairs_mutual_nearest_with_clear_margin():
    from app.services.error_mark_validation import normalize_error_mark_groups

    marks = [
        ErrorMark(mark_id=0, mark_type="cross", bbox=[0.30, 0.18, 0.36, 0.24], cross_bbox=[0.30, 0.18, 0.36, 0.24], confidence=0.95),
        ErrorMark(mark_id=1, mark_type="circle", bbox=[0.25, 0.22, 0.45, 0.42], circle_bbox=[0.25, 0.22, 0.45, 0.42], confidence=0.95),
        ErrorMark(mark_id=2, mark_type="circle", bbox=[0.70, 0.22, 0.90, 0.42], circle_bbox=[0.70, 0.22, 0.90, 0.42], confidence=0.95),
    ]

    groups, diagnostic = normalize_error_mark_groups(
        marks,
        dedup_iou_threshold=0.8,
        pair_max_distance_ratio=0.12,
        pair_max_relative_distance_ratio=1.0,
        pair_min_margin_ratio=0.2,
    )

    assert [group.mark_type for group in groups] == ["cross_circle", "circle"]
    assert diagnostic["pair_diagnostics"][0]["accepted"] is True
    assert diagnostic["pair_diagnostics"][0]["pair_tier"] == "strong"
    assert diagnostic["review_required_mark_ids"] == []


def test_normalize_error_mark_groups_marks_non_intersecting_pair_for_review():
    from app.services.error_mark_validation import normalize_error_mark_groups

    marks = [
        ErrorMark(
            mark_id=0,
            mark_type="cross",
            bbox=[0.30, 0.10, 0.35, 0.15],
            confidence=0.95,
        ),
        ErrorMark(
            mark_id=1,
            mark_type="circle",
            bbox=[0.30, 0.18, 0.50, 0.40],
            confidence=0.95,
        ),
    ]

    groups, diagnostic = normalize_error_mark_groups(
        marks,
        dedup_iou_threshold=0.8,
        pair_max_distance_ratio=0.12,
        pair_max_relative_distance_ratio=1.0,
        pair_min_margin_ratio=0.2,
    )

    assert [group.mark_type for group in groups] == ["cross_circle"]
    assert diagnostic["pair_diagnostics"][0]["pair_tier"] == "nearby_review"
    assert diagnostic["review_required_mark_ids"] == [0]


def test_group_red_evidence_regions_merges_transitive_bridge_components():
    from app.services.error_mark_validation import (
        RedMarkRegion,
        group_red_evidence_regions,
    )

    regions = [
        RedMarkRegion(
            bbox=[0.10, 0.10, 0.12, 0.12],
            pixel_count=10,
            area_ratio=0.0004,
            thinness_ratio=1.0,
        ),
        RedMarkRegion(
            bbox=[0.18, 0.10, 0.20, 0.12],
            pixel_count=10,
            area_ratio=0.0004,
            thinness_ratio=1.0,
        ),
        RedMarkRegion(
            bbox=[0.135, 0.10, 0.165, 0.12],
            pixel_count=10,
            area_ratio=0.0006,
            thinness_ratio=1.5,
        ),
    ]

    groups, diagnostic = group_red_evidence_regions(
        regions,
        max_gap_ratio=0.05,
        max_group_area_ratio=0.08,
    )

    assert len(groups) == 1
    assert groups[0].bbox == pytest.approx([0.10, 0.10, 0.20, 0.12])
    assert groups[0].pixel_count == 30
    assert diagnostic["evidence_group_count"] == 1


def test_normalize_error_mark_groups_rejects_relative_distance_and_ambiguous_margin():
    from app.services.error_mark_validation import normalize_error_mark_groups

    marks = [
        ErrorMark(mark_id=0, mark_type="cross", bbox=[0.30, 0.10, 0.32, 0.12], cross_bbox=[0.30, 0.10, 0.32, 0.12], confidence=0.95),
        ErrorMark(mark_id=1, mark_type="circle", bbox=[0.30, 0.16, 0.32, 0.18], circle_bbox=[0.30, 0.16, 0.32, 0.18], confidence=0.95),
        ErrorMark(mark_id=2, mark_type="circle", bbox=[0.33, 0.16, 0.35, 0.18], circle_bbox=[0.33, 0.16, 0.35, 0.18], confidence=0.95),
    ]

    groups, diagnostic = normalize_error_mark_groups(
        marks,
        dedup_iou_threshold=0.8,
        pair_max_distance_ratio=0.12,
        pair_max_relative_distance_ratio=1.0,
        pair_min_margin_ratio=0.2,
    )

    assert [group.mark_type for group in groups] == ["cross", "circle", "circle"]
    assert diagnostic["pair_diagnostics"][0]["accepted"] is False
    assert diagnostic["pair_diagnostics"][0]["reason"] in {
        "relative_distance_exceeded",
        "ambiguous_margin",
    }
    assert diagnostic["single_mark_group_count"] == 3


def test_expanded_mark_at_image_edge_is_clipped(tmp_path):
    from app.services.error_mark_validation import filter_valid_error_marks

    image_path = tmp_path / "edge.png"
    image = Image.new("RGB", (20, 20), "white")
    ImageDraw.Draw(image).rectangle((0, 0, 4, 4), fill=(230, 20, 20))
    image.save(image_path)

    valid, rejected, _diagnostics = filter_valid_error_marks(
        str(image_path),
        [_mark([0.0, 0.0, 0.2, 0.2])],
        confidence_threshold=0.85,
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.5,
    )

    assert [mark.mark_id for mark in valid] == [0]
    assert rejected == []


def test_localization_bbox_with_red_pixels_is_accepted(tmp_path):
    from app.services.error_mark_validation import (
        validate_localization_red_evidence,
    )

    image_path = tmp_path / "localization-red.png"
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 40, 40), fill=(220, 30, 30))
    image.save(image_path)

    diagnostic = validate_localization_red_evidence(
        str(image_path),
        bbox=[0.15, 0.15, 0.45, 0.45],
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.0,
    )

    assert diagnostic["accepted"] is True
    assert diagnostic["reason"] == "accepted"
    assert diagnostic["bbox"] == [0.15, 0.15, 0.45, 0.45]
    assert diagnostic["pixel_box"] == [15, 15, 45, 45]
    assert diagnostic["expansion_ratio"] == 0.0
    assert diagnostic["red_pixel_count"] > 0
    assert diagnostic["red_pixel_ratio"] >= 0.01


def test_localization_bbox_without_red_pixels_is_rejected(tmp_path):
    from app.services.error_mark_validation import (
        validate_localization_red_evidence,
    )

    image_path = tmp_path / "localization-white.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    diagnostic = validate_localization_red_evidence(
        str(image_path),
        bbox=[0.15, 0.15, 0.45, 0.45],
        red_pixel_min_ratio=0.01,
        expansion_ratio=0.05,
    )

    assert diagnostic["accepted"] is False
    assert diagnostic["reason"] == "insufficient_red_pixels"
    assert diagnostic["red_pixel_count"] == 0


def test_invalid_image_raises_safe_error(tmp_path):
    from app.services.error_mark_validation import (
        ErrorMarkImageInvalid,
        filter_valid_error_marks,
    )

    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"not-an-image")

    with pytest.raises(ErrorMarkImageInvalid):
        filter_valid_error_marks(
            str(image_path),
            [_mark()],
            confidence_threshold=0.85,
            red_pixel_min_ratio=0.01,
            expansion_ratio=0.05,
        )


def test_full_page_scan_keeps_red_marks_and_filters_grid_line_and_noise(tmp_path):
    from app.services.error_mark_validation import scan_red_mark_regions

    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    for y in range(30, 280, 40):
        draw.line((10, y, 390, y), fill=(225, 180, 180), width=1)
    draw.ellipse((40, 50, 100, 110), outline=(210, 25, 25), width=5)
    draw.line((180, 60, 230, 110), fill=(220, 20, 20), width=5)
    draw.line((230, 60, 180, 110), fill=(220, 20, 20), width=5)
    draw.line((20, 10, 380, 10), fill=(210, 25, 25), width=1)
    draw.point((350, 250), fill=(220, 20, 20))
    image.save(image_path)

    result = scan_red_mark_regions(
        str(image_path),
        max_edge=1600,
        min_component_pixels=12,
        max_component_area_ratio=0.08,
        max_thinness_ratio=18,
    )

    assert result.status == "detected"
    assert len(result.regions) == 2
    assert result.red_pixel_count > 0
    assert result.scanned_width == 400
    assert result.scanned_height == 300
    assert result.duration_ms >= 0


def test_full_page_scan_reports_no_reliable_red_marks(tmp_path):
    from app.services.error_mark_validation import scan_red_mark_regions

    image_path = tmp_path / "clean.png"
    Image.new("RGB", (120, 80), "white").save(image_path)

    result = scan_red_mark_regions(
        str(image_path), 1600, 12, 0.08, 18
    )

    assert result.status == "none"
    assert result.regions == []
    assert result.red_pixel_count == 0


def test_full_page_scan_handles_many_sparse_red_components_with_bounded_latency(
    tmp_path,
):
    """Regression for rescanning the whole mask once per red component."""
    from app.services.error_mark_validation import scan_red_mark_regions

    image_path = tmp_path / "sparse-red-components.png"
    image = Image.new("RGB", (1600, 1600), "white")
    draw = ImageDraw.Draw(image)
    for index in range(1600):
        x = 10 + (index % 40) * 39
        y = 10 + (index // 40) * 39
        draw.rectangle((x, y, x + 3, y + 3), fill=(220, 20, 20))
    image.save(image_path)

    result = scan_red_mark_regions(
        str(image_path),
        max_edge=1600,
        min_component_pixels=12,
        max_component_area_ratio=0.08,
        max_thinness_ratio=18,
    )

    assert len(result.regions) == 1600
    assert result.duration_ms < 2_000


def test_full_page_scan_rejects_invalid_image(tmp_path):
    from app.services.error_mark_validation import (
        ErrorMarkImageInvalid,
        scan_red_mark_regions,
    )

    image_path = tmp_path / "broken-scan.jpg"
    image_path.write_bytes(b"not-an-image")

    with pytest.raises(ErrorMarkImageInvalid):
        scan_red_mark_regions(str(image_path), 1600, 12, 0.08, 18)
