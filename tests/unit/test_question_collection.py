from types import SimpleNamespace


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


def test_recognized_item_not_auto_collected_is_retained_for_review():
    from app.tasks.process_image import collection_status_to_persist

    assert collection_status_to_persist("collected") == "collected"
    assert collection_status_to_persist("pending_review") == "pending_review"
    assert collection_status_to_persist("ignored") == "pending_review"


def test_strict_reprocessing_drops_automatically_ignored_candidates():
    from app.tasks.process_image import should_persist_candidate

    ignored = {"collection_status": "ignored"}
    pending = {"collection_status": "pending_review"}

    assert not should_persist_candidate(ignored, "false_positives")
    assert not should_persist_candidate(ignored, "both")
    assert should_persist_candidate(ignored, "missed_errors")
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
