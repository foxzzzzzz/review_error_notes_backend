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


def test_collection_reason_explains_answer_and_error_mark_conflict():
    from app.services.question_collection import collection_reason_for

    assert collection_reason_for(
        _values(
            raw_text="tiáo jiàn",
            answer="tiáo jiàn",
            reliable_mark=True,
        )
    ) == "答案与错误标记不一致"
