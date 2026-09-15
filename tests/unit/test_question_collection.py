from types import SimpleNamespace

import pytest


def _values(*, raw_text, answer, review_status="confirmed", reliable_mark=False):
    return SimpleNamespace(
        ocr_text=raw_text,
        ocr_answer=answer,
        review_status=review_status,
        crop_region={},
        reliable_error_mark=reliable_mark,
    )


def test_matching_answer_without_reliable_error_mark_is_ignored():
    from app.services.question_collection import collection_status_for

    assert collection_status_for(
        _values(raw_text="tiáo jiàn", answer="tiáo jiàn")
    ) == "ignored"


def test_mismatched_answer_with_reliable_error_mark_is_collected():
    from app.services.question_collection import collection_status_for

    assert collection_status_for(
        _values(
            raw_text="tiao jian",
            answer="tiáo jiàn",
            reliable_mark=True,
        )
    ) == "collected"


def test_unanswered_question_is_pending_review():
    from app.services.question_collection import collection_status_for

    assert collection_status_for(
        _values(raw_text="", answer="xìng yùn", reliable_mark=True)
    ) == "pending_review"


def test_uncertain_or_conflicting_item_requires_manual_review():
    from app.services.question_collection import collection_status_for

    assert collection_status_for(
        _values(
            raw_text="tiáo jiàn",
            answer="tiáo jiàn",
            reliable_mark=True,
        )
    ) == "pending_review"
    assert collection_status_for(
        _values(
            raw_text="tiao jian",
            answer="tiáo jiàn",
            review_status="needs_review",
            reliable_mark=True,
        )
    ) == "pending_review"


def test_pending_collection_keeps_image_in_review_state():
    from app.services.vision_recognition import image_status_for

    assert image_status_for([
        {"review_status": "confirmed", "collection_status": "pending_review"}
    ]) == "needs_review"


def test_ignored_item_does_not_keep_image_in_review_state():
    from app.services.vision_recognition import image_status_for

    assert image_status_for([
        {"review_status": "needs_review", "collection_status": "ignored"}
    ]) == "confirmed"


def test_ignored_item_is_not_unconditionally_converted_to_review():
    from app.services.question_collection import collection_status_for_decision
    from app.services.recognition_policy import CandidateDecision

    assert collection_status_for_decision(
        CandidateDecision(action="discard", reason="答案一致")
    ) is None


def test_evidence_policy_always_drops_ignored_candidates():
    from app.tasks.process_image import should_persist_candidate

    ignored = {"collection_status": "ignored"}
    pending = {"collection_status": "pending_review"}

    assert not should_persist_candidate(ignored, "false_positives")
    assert not should_persist_candidate(ignored, "both")
    assert not should_persist_candidate(ignored, "missed_errors")
    assert should_persist_candidate(pending, "false_positives")


def test_collection_reason_explains_answer_and_error_mark_conflict():
    from app.services.question_collection import collection_reason_for

    assert collection_reason_for(
        _values(
            raw_text="tiáo jiàn",
            answer="tiáo jiàn",
            reliable_mark=True,
        )
    ) == "答案与错误标记不一致"


def test_discards_pending_duplicate_when_same_image_has_collected_candidate():
    from app.tasks.process_image import discard_pending_duplicates_of_collected

    collected = {
        "collection_status": "collected",
        "ocr_text": "xiang qin",
        "ocr_answer": "相亲",
        "question_type": "write_word",
        "ocr_raw_json": {"instruction": "看拼音写词语", "prompt_text": "xiāng qīn"},
    }
    pending_duplicate = {
        **collected,
        "collection_status": "pending_review",
    }

    assert discard_pending_duplicates_of_collected(
        [collected, pending_duplicate]
    ) == [collected]


def test_task_discards_duplicates_after_recognition_before_persistence():
    from pathlib import Path

    source = (
        Path(__file__).parents[2] / "app" / "tasks" / "process_image.py"
    ).read_text(encoding="utf-8")

    recognition = source.index("result, question_values = recognize_question_batch(")
    deduplication = source.index(
        "question_values = discard_pending_duplicates_of_collected(question_values)"
    )
    persistence = source.index("for values in question_values")

    assert recognition < deduplication < persistence


def test_keeps_pending_candidate_when_its_content_differs_from_collected_candidate():
    from app.tasks.process_image import discard_pending_duplicates_of_collected

    collected = {
        "collection_status": "collected",
        "ocr_text": "xiang qin",
        "ocr_answer": "相亲",
        "question_type": "write_word",
        "ocr_raw_json": {"instruction": "看拼音写词语", "prompt_text": "xiāng qīn"},
    }
    pending_other_question = {
        **collected,
        "collection_status": "pending_review",
        "ocr_text": "zhan you",
        "ocr_answer": "战友",
        "ocr_raw_json": {"instruction": "看拼音写词语", "prompt_text": "zhàn yǒu"},
    }

    assert discard_pending_duplicates_of_collected(
        [collected, pending_other_question]
    ) == [collected, pending_other_question]


def _evidence_candidate(
    *,
    mark_id,
    geometry=None,
    text="相同学生文本",
    answer="相同建议答案",
):
    return {
        "recognition_pipeline": "chinese_marked_evidence_v1",
        "collection_status": "pending_review",
        "ocr_text": text,
        "ocr_answer": answer,
        "question_type": "write_word",
        "ocr_raw_json": {
            "evidence_bundle": {
                "identity": {
                    "image_id": "image-8",
                    "mark_id": mark_id,
                    "question_geometry": geometry
                    or {"bbox": [0.1, 0.2, 0.3, 0.4]},
                }
            }
        },
    }


def test_evidence_dedup_keeps_same_text_for_different_marks():
    from app.tasks.process_image import discard_pending_duplicates_of_collected

    first = _evidence_candidate(mark_id=1)
    second = _evidence_candidate(mark_id=2)

    assert discard_pending_duplicates_of_collected([first, second]) == [first, second]


def test_evidence_dedup_collapses_exact_identity_and_keeps_first():
    from app.tasks.process_image import discard_pending_duplicates_of_collected

    first = _evidence_candidate(mark_id=1, text="first")
    duplicate = _evidence_candidate(mark_id=1, text="changed", answer="changed")

    assert discard_pending_duplicates_of_collected([first, duplicate]) == [first]


def test_evidence_identity_canonicalizes_geometry_keys_and_numeric_forms():
    from app.tasks.process_image import candidate_identity_for

    first = _evidence_candidate(
        mark_id=1,
        geometry={
            "bbox": [-0.0, 0.1, 1, 0.4],
            "mark": {"confidence": 1.0, "mark_id": 1},
        },
    )
    reordered = _evidence_candidate(
        mark_id=1,
        geometry={
            "mark": {"mark_id": 1.0, "confidence": 1},
            "bbox": [0, 0.1, 1.0, 0.4],
        },
        text="unrelated text",
        answer="unrelated answer",
    )

    assert candidate_identity_for(first) == candidate_identity_for(reordered)


@pytest.mark.parametrize(
    "identity",
    [
        None,
        {},
        {
            "image_id": "",
            "mark_id": 1,
            "question_geometry": {"bbox": [0.1, 0.2, 0.3, 0.4]},
        },
        {
            "image_id": "   ",
            "mark_id": 1,
            "question_geometry": {"bbox": [0.1, 0.2, 0.3, 0.4]},
        },
        {
            "image_id": "image-8",
            "mark_id": -1,
            "question_geometry": {"bbox": [0.1, 0.2, 0.3, 0.4]},
        },
        {"image_id": "image-8", "mark_id": 1},
        {
            "image_id": "image-8",
            "mark_id": True,
            "question_geometry": {"bbox": [0.1, 0.2, 0.3, 0.4]},
        },
        {
            "image_id": "image-8",
            "mark_id": 1,
            "question_geometry": {"bbox": [float("nan"), 0.2, 0.3, 0.4]},
        },
        {
            "image_id": "image-8",
            "mark_id": 1,
            "question_geometry": {"question_bbox": [0.1, True, 0.3, 0.4]},
        },
        {
            "image_id": "image-8",
            "mark_id": 1,
            "question_geometry": {"answer_bbox": [-0.1, 0.2, 0.3, 0.4]},
        },
        {
            "image_id": "image-8",
            "mark_id": 1,
            "question_geometry": {"prompt_bbox": [0.3, 0.2, 0.3, 0.4]},
        },
        {
            "image_id": "image-8",
            "mark_id": 1,
            "question_geometry": {
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "mark": {"bbox": [0.1, 0.2, 1.1, 0.4]},
            },
        },
    ],
)
def test_malformed_evidence_identity_never_falls_back_to_text(identity):
    from app.tasks.process_image import candidate_identity_for

    candidate = _evidence_candidate(mark_id=1)
    candidate["ocr_raw_json"]["evidence_bundle"]["identity"] = identity

    with pytest.raises(ValueError, match="evidence identity"):
        candidate_identity_for(candidate)


def test_evidence_identity_allows_diagnostic_booleans_with_valid_bboxes():
    from app.tasks.process_image import candidate_identity_for

    candidate = _evidence_candidate(
        mark_id=1,
        geometry={
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "question_bbox": [0.1, 0.2, 0.3, 0.4],
            "answer_bbox": None,
            "prompt_bbox": [0.1, 0.2, 0.2, 0.3],
            "mark": {
                "bbox": [0.15, 0.25, 0.2, 0.3],
                "cross_bbox": None,
                "circle_bbox": [0.14, 0.24, 0.21, 0.31],
            },
            "localization": {"geometry_diagnostic": {"passed": True}},
        },
    )

    identity = candidate_identity_for(candidate)

    assert identity[0] == "chinese_marked_evidence_v1"
