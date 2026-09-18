from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def _primary_candidate(index=0):
    return {
        "recognition_pipeline": "chinese_marked_evidence_v1",
        "mark_status": "needs_review",
        "answer_status": "unresolved",
        "collection_status": "pending_review",
        "crop_region": {
            "model_bbox": [0.2, 0.2, 0.4, 0.4],
            "display_bbox": [0.1, 0.1, 0.5, 0.5],
            "index": index,
        },
        "ocr_raw_json": {"page_review_reasons": ["primary_page_recognition_pending_review"]},
    }


@pytest.mark.parametrize(
    ("candidate_count", "regions", "expected_covered", "expected_uncovered"),
    [
        (1, [[0.25, 0.25, 0.3, 0.3]], [0], []),
        (1, [], [], []),
        (0, [[0.25, 0.25, 0.3, 0.3]], [], [0]),
        (0, [], [], []),
    ],
    ids=["model_and_cv", "model_without_cv", "cv_without_model", "neither"],
)
def test_primary_page_cv_audit_is_advisory_and_never_changes_candidates(
    candidate_count, regions, expected_covered, expected_uncovered
):
    """CV may flag coverage gaps, but it cannot decide primary-page identity."""
    from app.services.chinese_marked_evidence import audit_primary_page_cv_coverage

    candidates = [_primary_candidate(index) for index in range(candidate_count)]
    result = audit_primary_page_cv_coverage(candidates, regions)

    assert len(candidates) == candidate_count
    assert result["candidate_count"] == candidate_count
    assert result["covered_region_indexes"] == expected_covered
    assert result["uncovered_region_indexes"] == expected_uncovered
    assert result["requires_manual_review"] is bool(expected_uncovered)
    assert all(
        candidate["answer_status"] != "confirmed"
        and candidate["collection_status"] != "collected"
        for candidate in candidates
    )
    if candidate_count:
        assert candidates[0]["ocr_raw_json"]["local_cv_audit"]["candidate_bbox"] == [
            0.2,
            0.2,
            0.4,
            0.4,
        ]


def test_primary_page_cv_audit_uses_model_bbox_not_display_bbox():
    from app.services.chinese_marked_evidence import audit_primary_page_cv_coverage

    candidate = _primary_candidate()
    result = audit_primary_page_cv_coverage(
        [candidate], [[0.12, 0.12, 0.18, 0.18]]
    )

    assert result["covered_region_indexes"] == []
    assert result["uncovered_region_indexes"] == [0]


def test_legacy_question_remains_downstream_eligible():
    from app.services.chinese_marked_evidence import is_downstream_eligible

    question = SimpleNamespace(recognition_pipeline=None, answer_status=None)

    assert is_downstream_eligible(question)


def test_other_pipeline_question_remains_downstream_eligible():
    from app.services.chinese_marked_evidence import is_downstream_eligible

    question = SimpleNamespace(
        recognition_pipeline="legacy_ocr_v2",
        answer_status="suggested",
    )

    assert is_downstream_eligible(question)


@pytest.mark.parametrize("status", ["suggested", "unresolved", None])
def test_new_pipeline_requires_confirmed_answer(status):
    from app.services.chinese_marked_evidence import (
        PIPELINE_NAME,
        is_downstream_eligible,
    )

    question = SimpleNamespace(
        recognition_pipeline=PIPELINE_NAME,
        answer_status=status,
    )

    assert not is_downstream_eligible(question)


def test_new_pipeline_confirmed_answer_is_eligible():
    from app.services.chinese_marked_evidence import (
        PIPELINE_NAME,
        is_downstream_eligible,
    )

    question = SimpleNamespace(
        recognition_pipeline=PIPELINE_NAME,
        answer_status="confirmed",
    )

    assert is_downstream_eligible(question)


def _observation(
    *, source, text, bbox, source_class="printed", confidence=0.9, role=None
):
    """Create a hand-authored provider observation for evidence tests."""
    from app.services.chinese_marked_evidence import EvidenceObservation

    values = dict(
        source=source,
        text=text,
        bbox=bbox,
        source_class=source_class,
        confidence=confidence,
    )
    if role is not None:
        values["role"] = role
    return EvidenceObservation(**values)


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": "value"},
        {"bbox": [0.4, 0.2, 0.1, 0.5]},
        {"confidence": 1.01},
        {"source_class": "machine_guess"},
    ],
)
def test_evidence_observation_rejects_unknown_or_invalid_evidence(payload):
    """A DTO change that permits unchecked provider evidence must fail here."""
    from app.services.chinese_marked_evidence import EvidenceObservation

    base = {
        "source": "deepseek",
        "text": "金鱼",
        "bbox": [0.1, 0.2, 0.4, 0.5],
        "source_class": "student_handwriting",
        "confidence": 0.9,
    }

    with pytest.raises(ValidationError):
        EvidenceObservation(**(base | payload))


def test_answer_cell_text_stays_student_handwriting():
    """Changing spatial-role routing to text heuristics must fail this test."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i1",
        mark_id=7,
        question_geometry={"bbox": [0.1, 0.2, 0.4, 0.5]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="金鱼",
                bbox=[0.2, 0.3, 0.3, 0.4],
                source_class="student_handwriting",
            )
        ],
        ocr_observations=[
            _observation(
                source="ocr",
                text="金鱼",
                bbox=[0.2, 0.3, 0.3, 0.4],
                source_class="student_handwriting",
            )
        ],
        structure_observation={"student_answer_bbox": [0.2, 0.3, 0.3, 0.4]},
    )

    assert bundle.fields["student_answer"].selected_value == "金鱼"
    assert bundle.answer_suggestion.status == "unresolved"
    assert all(
        observation.source_class != "printed"
        for observation in bundle.fields["student_answer"].observations
    )


def test_answer_cell_role_outranks_overlapping_printed_prompt_geometry():
    """Routing an answer cell through broad prompt geometry must fail this test."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i1-overlap",
        mark_id=70,
        question_geometry={"bbox": [0.1, 0.2, 0.4, 0.5]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="常见词",
                bbox=[0.2, 0.3, 0.3, 0.4],
            )
        ],
        ocr_observations=[],
        structure_observation={
            "printed_prompt_bbox": [0.1, 0.2, 0.4, 0.5],
            "student_answer_bbox": [0.2, 0.3, 0.3, 0.4],
        },
    )

    assert bundle.fields["student_answer"].selected_value == "常见词"
    assert bundle.fields["student_answer"].observations[0].source_class == (
        "student_handwriting"
    )


def test_independent_matching_prompt_evidence_is_confirmed():
    """Dropping either independent source must prevent confirmation."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i2",
        mark_id=8,
        question_geometry={"bbox": [0.1, 0.2, 0.4, 0.5]},
        deepseek_observations=[
            _observation(
                source="deepseek", text="选择词语", bbox=[0.1, 0.2, 0.4, 0.3]
            )
        ],
        ocr_observations=[
            _observation(source="ocr", text="选择词语", bbox=[0.1, 0.2, 0.4, 0.3])
        ],
        structure_observation={"printed_prompt_bbox": [0.1, 0.2, 0.4, 0.3]},
    )

    assert bundle.fields["printed_prompt"].status == "confirmed"
    assert bundle.question_evidence_status == "confirmed"


def test_single_prompt_source_is_supported_not_confirmed():
    """Treating one provider as independent agreement must fail this test."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i3",
        mark_id=9,
        question_geometry={"bbox": [0.1, 0.2, 0.4, 0.5]},
        deepseek_observations=[
            _observation(source="deepseek", text="选择词语", bbox=[0.1, 0.2, 0.4, 0.3])
        ],
        ocr_observations=[],
        structure_observation={"printed_prompt_bbox": [0.1, 0.2, 0.4, 0.3]},
    )

    assert bundle.fields["printed_prompt"].status == "supported"
    assert bundle.question_evidence_status == "supported"


def test_conflicting_printed_prompt_preserves_both_sources():
    """Collapsing distinct provider values must fail this test."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i4",
        mark_id=10,
        question_geometry={"bbox": [0.1, 0.2, 0.4, 0.5]},
        deepseek_observations=[
            _observation(source="deepseek", text="选择词语", bbox=[0.1, 0.2, 0.4, 0.3])
        ],
        ocr_observations=[
            _observation(source="ocr", text="填写词语", bbox=[0.1, 0.2, 0.4, 0.3])
        ],
        structure_observation={"printed_prompt_bbox": [0.1, 0.2, 0.4, 0.3]},
    )

    assert bundle.question_evidence_status == "conflict"
    assert len(bundle.fields["printed_prompt"].observations) == 2
    assert bundle.fields["printed_prompt"].conflicts == ["填写词语", "选择词语"]


def test_absent_evidence_is_insufficient_and_answer_is_unresolved():
    """Inventing a value for absent evidence must fail this test."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i5",
        mark_id=11,
        question_geometry={"bbox": [0.1, 0.2, 0.4, 0.5]},
        deepseek_observations=[],
        ocr_observations=[],
        structure_observation={},
    )

    assert bundle.question_evidence_status == "insufficient"
    assert bundle.answer_suggestion.status == "unresolved"
    assert bundle.identity.image_id == "i5"
    assert bundle.identity.mark_id == 11
    assert bundle.identity.question_geometry == {"bbox": [0.1, 0.2, 0.4, 0.5]}


def test_answer_status_cannot_be_confirmed_before_human_review():
    """Adding a confirmed automatic answer state must fail this test."""
    from app.services.chinese_marked_evidence import AnswerSuggestion

    with pytest.raises(ValidationError):
        AnswerSuggestion(status="confirmed", selected_value="金鱼")


def test_pending_values_are_fixed_to_manual_review_and_include_bundle():
    """Caller-controlled collection status or a confirmed answer must fail here."""
    from app.services.chinese_marked_evidence import (
        assemble_question_evidence,
        build_pending_evidence_values,
    )

    bundle = assemble_question_evidence(
        image_id="i6",
        mark_id=12,
        question_geometry={"bbox": [0.1, 0.2, 0.4, 0.5]},
        deepseek_observations=[],
        ocr_observations=[],
        structure_observation={},
        suggested_answer_candidate="金鱼",
    )

    values = build_pending_evidence_values(
        bundle,
        student_text="学生原文",
        question_type="词语辨析",
        crop_region={"bbox": [0.1, 0.2, 0.4, 0.5]},
    )

    assert values["recognition_pipeline"] == "chinese_marked_evidence_v1"
    assert values["collection_status"] == "pending_review"
    assert values["review_status"] == "needs_review"
    assert values["answer_status"] == "suggested"
    assert values["ocr_text"] == "学生原文"
    assert values["ocr_answer"] == "金鱼"
    assert values["question_type"] == "词语辨析"
    assert values["crop_region"] == {"bbox": [0.1, 0.2, 0.4, 0.5]}
    assert values["ocr_raw_json"]["evidence_bundle"]["identity"]["mark_id"] == 12


def test_page_primary_evidence_keeps_model_bbox_separate_from_display_bbox():
    from app.services.chinese_marked_evidence import (
        assemble_question_evidence,
        build_pending_evidence_values,
    )

    model_bbox = [0.4, 0.4, 0.6, 0.6]
    display_bbox = [0.3, 0.3, 0.7, 0.7]
    bundle = assemble_question_evidence(
        image_id="page-primary",
        mark_id=0,
        question_geometry={"bbox": model_bbox, "source": "deepseek_page_primary"},
        deepseek_observations=[
            _observation(source="deepseek", text="题干", bbox=model_bbox)
        ],
        ocr_observations=[],
        structure_observation={},
    )
    values = build_pending_evidence_values(
        bundle,
        student_text=None,
        question_type=None,
        crop_region={
            "bbox": display_bbox,
            "model_bbox": model_bbox,
            "display_bbox": display_bbox,
            "display_bbox_scale": 2.0,
            "bbox_format": "normalized_ltrb",
        },
        raw_evidence={
            "model_bbox": model_bbox,
            "display_bbox": display_bbox,
            "display_bbox_scale": 2.0,
        },
    )

    assert values["ocr_raw_json"]["evidence_bundle"]["identity"]["question_geometry"]["bbox"] == model_bbox
    assert values["ocr_raw_json"]["model_bbox"] == model_bbox
    assert values["crop_region"]["bbox"] == display_bbox
    assert values["crop_region"]["model_bbox"] == model_bbox


def test_confirmed_requires_each_provider_observation_inside_field_structure():
    """Confirming an observation outside its role geometry must fail this test."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i-confirmed-geometry",
        mark_id=13,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(source="deepseek", text="题干", bbox=[0.1, 0.1, 0.3, 0.2])
        ],
        ocr_observations=[
            _observation(source="ocr", text="题干", bbox=[0.6, 0.6, 0.8, 0.8])
        ],
        structure_observation={"printed_prompt_bbox": [0.1, 0.1, 0.3, 0.2]},
    )

    assert bundle.fields["printed_prompt"].status == "supported"
    assert bundle.question_evidence_status == "supported"


def test_shared_bbox_edge_does_not_route_printed_text_to_answer_cell():
    """Counting a shared edge as an answer-cell overlap must fail this test."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i-edge",
        mark_id=14,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(source="deepseek", text="题干", bbox=[0.1, 0.1, 0.2, 0.3])
        ],
        ocr_observations=[],
        structure_observation={"student_answer_bbox": [0.2, 0.3, 0.4, 0.5]},
    )

    assert bundle.fields["student_answer"].observations == []
    assert bundle.fields["printed_prompt"].selected_value == "题干"


@pytest.mark.parametrize(
    "bbox",
    [
        [float("nan"), 0.1, 0.2, 0.3],
        [0.1, 0.1, float("inf"), 0.3],
        [0.1, 0.1, 0.1, 0.3],
        [0.1, 0.1, 0.3, 0.1],
    ],
)
def test_observation_rejects_nonfinite_or_degenerate_bbox(bbox):
    """Allowing invalid observation geometry must fail this test."""
    from app.services.chinese_marked_evidence import EvidenceObservation

    with pytest.raises(ValidationError):
        EvidenceObservation(
            source="deepseek",
            text="金鱼",
            bbox=bbox,
            source_class="student_handwriting",
            confidence=0.9,
        )


def test_structure_rejects_nonfinite_or_degenerate_bbox():
    """Ignoring invalid declared field geometry must fail this test."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    with pytest.raises(ValueError):
        assemble_question_evidence(
            image_id="i-structure",
            mark_id=15,
            question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
            deepseek_observations=[],
            ocr_observations=[],
            structure_observation={"student_answer_bbox": [0.1, 0.1, 0.1, 0.3]},
        )


@pytest.mark.parametrize(
    "deepseek_observations, ocr_observations",
    [
        ([_observation(source="ocr", text="题干", bbox=[0.1, 0.1, 0.3, 0.2])], []),
        ([], [_observation(source="deepseek", text="题干", bbox=[0.1, 0.1, 0.3, 0.2])]),
    ],
)
def test_provider_lists_reject_mismatched_observation_source(
    deepseek_observations, ocr_observations
):
    """Accepting a provider observation in the wrong channel must fail here."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    with pytest.raises(ValueError, match="source"):
        assemble_question_evidence(
            image_id="i-provider",
            mark_id=16,
            question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
            deepseek_observations=deepseek_observations,
            ocr_observations=ocr_observations,
            structure_observation={"printed_prompt_bbox": [0.1, 0.1, 0.3, 0.2]},
        )


def test_student_handwriting_does_not_create_an_answer_suggestion():
    """Promoting a student transcription to a correct-answer candidate must fail."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i-student-only",
        mark_id=17,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="学生填写",
                bbox=[0.2, 0.3, 0.3, 0.4],
                source_class="student_handwriting",
            )
        ],
        ocr_observations=[],
        structure_observation={"student_answer_bbox": [0.2, 0.3, 0.3, 0.4]},
    )

    assert bundle.fields["student_answer"].selected_value == "学生填写"
    assert bundle.answer_suggestion.status == "unresolved"
    assert bundle.answer_suggestion.selected_value is None


def test_independent_answer_candidate_controls_bundle_and_pending_value():
    """Allowing persistence to receive a different answer than the bundle must fail."""
    from app.services.chinese_marked_evidence import (
        assemble_question_evidence,
        build_pending_evidence_values,
    )

    bundle = assemble_question_evidence(
        image_id="i-candidate",
        mark_id=18,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[],
        ocr_observations=[],
        structure_observation={},
        suggested_answer_candidate="正确答案",
    )
    values = build_pending_evidence_values(
        bundle,
        student_text="学生填写",
        question_type="词语辨析",
        crop_region={"bbox": [0.0, 0.0, 1.0, 1.0]},
    )

    assert bundle.answer_suggestion.status == "suggested"
    assert bundle.answer_suggestion.selected_value == "正确答案"
    assert values["answer_status"] == "suggested"
    assert values["ocr_answer"] == "正确答案"
    with pytest.raises(TypeError):
        build_pending_evidence_values(
            bundle,
            student_text="学生填写",
            suggested_answer="不同答案",
            question_type="词语辨析",
            crop_region={"bbox": [0.0, 0.0, 1.0, 1.0]},
        )


def test_mark_status_is_upstream_or_needs_review_never_suggested():
    """Reintroducing a suggested mark state must fail this test."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    unresolved_mark = assemble_question_evidence(
        image_id="i-mark-default",
        mark_id=19,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[],
        ocr_observations=[],
        structure_observation={},
    )
    confirmed_mark = assemble_question_evidence(
        image_id="i-mark-confirmed",
        mark_id=20,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[],
        ocr_observations=[],
        structure_observation={},
        mark_status="confirmed",
    )
    unknown_mark = assemble_question_evidence(
        image_id="i-mark-unknown",
        mark_id=21,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[],
        ocr_observations=[],
        structure_observation={},
        mark_status="unknown",
    )

    assert unresolved_mark.mark_status == "needs_review"
    assert confirmed_mark.mark_status == "confirmed"
    assert unknown_mark.mark_status == "needs_review"


def test_teacher_correction_overlapping_answer_cell_remains_independent():
    """Routing a declared teacher correction into the student answer must fail."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i-teacher",
        mark_id=22,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="老师批注",
                bbox=[0.2, 0.3, 0.3, 0.4],
                source_class="teacher_correction",
            )
        ],
        ocr_observations=[],
        structure_observation={"student_answer_bbox": [0.2, 0.3, 0.3, 0.4]},
    )

    assert bundle.fields["student_answer"].observations == []
    assert bundle.fields["teacher_correction"].selected_value == "老师批注"
    assert bundle.answer_suggestion.status == "unresolved"


def test_declared_student_handwriting_cannot_be_upgraded_by_broad_prompt_bbox():
    """A broad prompt box must not turn declared handwriting into confirmed print."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="i-handwriting-declaration",
        mark_id=23,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="学生填写",
                bbox=[0.2, 0.3, 0.3, 0.4],
                source_class="student_handwriting",
            )
        ],
        ocr_observations=[
            _observation(
                source="ocr",
                text="学生填写",
                bbox=[0.2, 0.3, 0.3, 0.4],
                source_class="student_handwriting",
            )
        ],
        structure_observation={"printed_prompt_bbox": [0.0, 0.0, 1.0, 1.0]},
    )

    assert bundle.question_evidence_status != "confirmed"
    assert len(bundle.fields["student_answer"].observations) == 2
    assert bundle.fields["printed_prompt"].observations == []


def test_explicit_printed_roles_remain_separate_and_optional_absence_is_neutral():
    """Collapsing printed roles into one prompt would create false field conflicts."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="role-separation",
        mark_id=24,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="看拼音写词语",
                bbox=[0.1, 0.1, 0.8, 0.2],
                source_class="printed",
                role="printed_instruction",
            ),
            _observation(
                source="deepseek",
                text="yǎn jìng",
                bbox=[0.1, 0.3, 0.8, 0.4],
                source_class="printed",
                role="printed_pinyin",
            ),
        ],
        ocr_observations=[],
        structure_observation={},
    )

    assert bundle.fields["printed_instruction"].selected_value == "看拼音写词语"
    assert bundle.fields["printed_pinyin"].selected_value == "yǎn jìng"
    assert bundle.fields["printed_prompt"].status == "insufficient"
    assert bundle.question_evidence_status == "supported"


def test_any_populated_printed_role_conflict_controls_question_status():
    """Ignoring a conflict outside printed_prompt would hide auditable disagreement."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="role-conflict",
        mark_id=25,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="yǎn jìng",
                bbox=[0.1, 0.3, 0.8, 0.4],
                source_class="printed",
                role="printed_pinyin",
            ),
        ],
        ocr_observations=[
            _observation(
                source="ocr",
                text="yǎn jin",
                bbox=[0.1, 0.3, 0.8, 0.4],
                source_class="unknown",
            ),
        ],
        structure_observation={"printed_pinyin_bbox": [0.1, 0.3, 0.8, 0.4]},
    )

    assert bundle.fields["printed_pinyin"].status == "conflict"
    assert bundle.question_evidence_status == "conflict"


def test_narrow_printed_role_routes_ocr_inside_broad_prompt_geometry():
    """First-match routing would collapse pinyin OCR into a broad prompt field."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="narrow-role",
        mark_id=27,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="yǎn jìng",
                bbox=[0.2, 0.3, 0.7, 0.4],
                source_class="printed",
                role="printed_pinyin",
            )
        ],
        ocr_observations=[
            _observation(
                source="ocr",
                text="yǎn jìng",
                bbox=[0.2, 0.3, 0.7, 0.4],
                source_class="unknown",
            )
        ],
        structure_observation={
            "printed_prompt_bbox": [0.1, 0.1, 0.9, 0.5],
            "printed_pinyin_bbox": [0.2, 0.3, 0.7, 0.4],
        },
    )

    assert bundle.fields["printed_pinyin"].status == "confirmed"
    assert bundle.fields["printed_prompt"].status == "insufficient"
    assert bundle.question_evidence_status == "confirmed"


def test_role_routing_hint_routes_unknown_ocr_without_confirming():
    """Routing geometry may group OCR with pinyin, but cannot independently confirm it."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="routing-only",
        mark_id=30,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="yǎn jìng",
                bbox=[0.2, 0.3, 0.7, 0.4],
                source_class="printed",
                role="printed_pinyin",
            )
        ],
        ocr_observations=[
            _observation(
                source="ocr",
                text="yǎn jìng",
                bbox=[0.2, 0.3, 0.7, 0.4],
                source_class="unknown",
            )
        ],
        structure_observation={},
        role_routing_hints={"printed_pinyin": [0.2, 0.3, 0.7, 0.4]},
    )

    field = bundle.fields["printed_pinyin"]
    assert field.status == "supported"
    assert [observation.source for observation in field.observations] == [
        "deepseek",
        "ocr",
    ]
    assert field.observations[1].role == "unknown"
    assert bundle.question_evidence_status == "supported"


def test_narrowest_role_routing_hint_wins_inside_overlap():
    """A unique smaller hint must route OCR without depending on mapping order."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="routing-overlap",
        mark_id=31,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[],
        ocr_observations=[
            _observation(
                source="ocr",
                text="yǎn jìng",
                bbox=[0.3, 0.3, 0.6, 0.4],
                source_class="unknown",
            )
        ],
        structure_observation={},
        role_routing_hints={
            "printed_prompt": [0.1, 0.1, 0.9, 0.5],
            "printed_pinyin": [0.2, 0.25, 0.7, 0.45],
        },
    )

    assert len(bundle.fields["printed_pinyin"].observations) == 1
    assert bundle.fields["printed_prompt"].observations == []


def test_equal_area_role_routing_hints_are_ambiguous_and_order_independent():
    """Equal-area hint ambiguity must remain unknown instead of using dict order."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    routed_fields = []
    for hints in (
        {
            "printed_instruction": [0.1, 0.1, 0.8, 0.2],
            "printed_pinyin": [0.15, 0.1, 0.85, 0.2],
        },
        {
            "printed_pinyin": [0.15, 0.1, 0.85, 0.2],
            "printed_instruction": [0.1, 0.1, 0.8, 0.2],
        },
    ):
        bundle = assemble_question_evidence(
            image_id="routing-ambiguous",
            mark_id=32,
            question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
            deepseek_observations=[],
            ocr_observations=[
                _observation(
                    source="ocr",
                    text="重叠文本",
                    bbox=[0.2, 0.12, 0.7, 0.18],
                    source_class="unknown",
                )
            ],
            structure_observation={},
            role_routing_hints=hints,
        )
        routed_fields.append(
            [
                field_name
                for field_name, field in bundle.fields.items()
                if field.observations
            ]
        )

    assert routed_fields == [["unknown"], ["unknown"]]


@pytest.mark.parametrize(
    "role_routing_hints",
    [
        {"printed_pinyin": [0.1, 0.1, 0.1, 0.2]},
        {"printed_pinyin": [0.1, 0.1, float("nan"), 0.2]},
        {"student_answer": [0.1, 0.1, 0.2, 0.2]},
        {1: [0.1, 0.1, 0.2, 0.2]},
    ],
)
def test_invalid_role_routing_hints_are_rejected(role_routing_hints):
    """Routing hints must be valid printed-role rectangles only."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    with pytest.raises(ValueError, match="role_routing_hints"):
        assemble_question_evidence(
            image_id="invalid-routing",
            mark_id=33,
            question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
            deepseek_observations=[],
            ocr_observations=[],
            structure_observation={},
            role_routing_hints=role_routing_hints,
        )


def test_evidence_payload_has_schema_version_and_compatible_metadata():
    """Dropping schema or persistence metadata would break later review consumers."""
    from app.services.chinese_marked_evidence import (
        assemble_question_evidence,
        build_pending_evidence_values,
    )

    bundle = assemble_question_evidence(
        image_id="schema-version",
        mark_id=26,
        question_geometry={"bbox": [0.1, 0.2, 0.4, 0.5]},
        deepseek_observations=[],
        ocr_observations=[],
        structure_observation={},
    )
    values = build_pending_evidence_values(
        bundle,
        student_text=None,
        question_type="write_word",
        crop_region={"bbox": [0.1, 0.2, 0.4, 0.5]},
        subject="chinese",
        tags=["词语"],
        difficulty=2,
        raw_evidence={
            "evidence_bundle": {"schema_version": "forged"},
            "content_diagnostic": {"status": "missing"},
        },
    )

    assert values["subject"] == "chinese"
    assert values["tags"] == ["词语"]
    assert values["difficulty"] == 2
    assert values["ocr_raw_json"]["evidence_bundle"]["schema_version"] == 1
    assert values["ocr_raw_json"]["content_diagnostic"] == {"status": "missing"}


def test_blank_printed_observation_is_retained_but_overall_is_insufficient():
    """Counting an unusable blank observation as populated would false-support it."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="blank-only",
        mark_id=28,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="   ",
                bbox=[0.1, 0.1, 0.8, 0.2],
                source_class="printed",
                role="printed_instruction",
            )
        ],
        ocr_observations=[],
        structure_observation={},
    )

    assert len(bundle.fields["printed_instruction"].observations) == 1
    assert bundle.fields["printed_instruction"].status == "insufficient"
    assert bundle.question_evidence_status == "insufficient"


def test_confirmed_printed_field_ignores_blank_optional_field_for_overall_status():
    """A blank optional role must not downgrade independently confirmed print."""
    from app.services.chinese_marked_evidence import assemble_question_evidence

    bundle = assemble_question_evidence(
        image_id="blank-optional",
        mark_id=29,
        question_geometry={"bbox": [0.0, 0.0, 1.0, 1.0]},
        deepseek_observations=[
            _observation(
                source="deepseek",
                text="选择词语",
                bbox=[0.1, 0.1, 0.8, 0.2],
                source_class="printed",
                role="printed_prompt",
            ),
            _observation(
                source="deepseek",
                text="\t",
                bbox=[0.2, 0.3, 0.7, 0.4],
                source_class="printed",
                role="printed_pinyin",
            ),
        ],
        ocr_observations=[
            _observation(
                source="ocr",
                text="选择词语",
                bbox=[0.1, 0.1, 0.8, 0.2],
                source_class="unknown",
            )
        ],
        structure_observation={"printed_prompt_bbox": [0.1, 0.1, 0.8, 0.2]},
    )

    assert bundle.fields["printed_prompt"].status == "confirmed"
    assert bundle.fields["printed_pinyin"].status == "insufficient"
    assert bundle.question_evidence_status == "confirmed"


def test_explicit_printed_role_inside_answer_bbox_is_preserved_as_role_conflict():
    from app.services.chinese_marked_evidence import assemble_question_evidence

    observation = _observation(
        source="deepseek",
        text="学生写的字",
        bbox=[0.2, 0.3, 0.3, 0.4],
        source_class="printed",
        role="printed_prompt",
    )

    bundle = assemble_question_evidence(
        image_id="image-1",
        mark_id=1,
        question_geometry={"bbox": [0.1, 0.1, 0.5, 0.5]},
        deepseek_observations=[observation],
        ocr_observations=[],
        structure_observation={"student_answer_bbox": [0.15, 0.25, 0.35, 0.45]},
    )

    assert bundle.fields["printed_prompt"].selected_value is None
    assert bundle.fields["printed_prompt"].observations == []
    assert bundle.fields["unknown"].observations == [observation]
    assert [
        conflict.model_dump(mode="json") for conflict in bundle.role_conflicts
    ] == [
        {
            "claimed_role": "printed_prompt",
            "conflicting_role": "student_answer",
            "reason": "printed_observation_overlaps_student_answer_bbox",
            "observation": observation.model_dump(mode="json"),
        }
    ]
    assert bundle.question_evidence_status == "conflict"
