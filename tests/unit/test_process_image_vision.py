from app.services.vision_recognition import ErrorMark, LocalizationItem, VisionItem


def test_worker_registers_complete_foreign_key_model_graph():
    import ast
    from pathlib import Path

    source = (Path(__file__).parents[2] / "app" / "tasks" / "process_image.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_models = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app.models.")
    }

    assert {
        "app.models.student",
        "app.models.wrong_image",
        "app.models.wrong_question",
        "app.models.practice_sheet",
        "app.models.sheet_item",
    } <= imported_models


def _item(**overrides):
    data = {
        "raw_text": "qin tin\n蜻蜓",
        "instruction": "看词语写拼音",
        "prompt_text": "蜻蜓",
        "normalized_text": "qīng tíng\n蜻蜓",
        "answer": "qīng tíng",
        "subject": "chinese",
        "question_type": "write_pinyin",
        "tags": ["拼音"],
        "difficulty": 2,
        "confidence": 0.92,
        "uncertain_segments": [],
    }
    data.update(overrides)
    return VisionItem(**data)


def _mark():
    return ErrorMark(
        mark_id=0,
        mark_type="circle",
        bbox=[0.2, 0.25, 0.3, 0.35],
        confidence=0.96,
    )


def test_question_values_preserve_raw_writing_and_normalized_content():
    from app.services.vision_recognition import build_question_values

    localization = LocalizationItem(
        index=0,
        matched=True,
        mark_ids=[0],
        bbox=[0.05, 0.15, 0.45, 0.5],
        observed_prompt_text="蜻蜓",
        observed_raw_text="qin tin\n蜻蜓",
        confidence=0.93,
    )
    values = build_question_values(
        _item(),
        index=0,
        confidence_threshold=0.85,
        localization=localization,
        localization_threshold=0.85,
        localization_max_area_ratio=0.35,
        marks={0: _mark()},
        normalized_tags=["拼音", "老师批改"],
    )

    assert values["ocr_text"] == "qin tin\n蜻蜓"
    assert values["ocr_text"] != values["ocr_raw_json"]["normalized_text"]
    assert values["ocr_answer"] == "qīng tíng"
    assert values["ocr_raw_json"]["provider"] == "minimax"
    assert values["ocr_raw_json"]["confidence"] == 0.92
    assert values["ocr_raw_json"]["raw_text"] == values["ocr_text"]
    assert values["ocr_raw_json"]["instruction"] == "看词语写拼音"
    assert values["ocr_raw_json"]["prompt_text"] == "蜻蜓"
    assert values["ocr_raw_json"]["answer"] == values["ocr_answer"]
    assert values["ocr_raw_json"]["subject"] == "chinese"
    assert "bbox" not in values["ocr_raw_json"]
    assert values["crop_region"] == {
        "bbox": [0.05, 0.15, 0.45, 0.5],
        "bbox_format": "normalized_ltrb",
        "bbox_source": "minimax_marker_anchored",
        "bbox_confidence": 0.93,
        "localization_status": "verified",
        "mark_ids": [0],
        "index": 0,
    }
    assert values["tags"] == ["拼音", "老师批改"]
    assert values["status"] == "confirmed"


def test_low_confidence_localization_discards_candidate_bbox():
    from app.services.vision_recognition import build_question_values

    localization = LocalizationItem(
        index=1,
        matched=True,
        mark_ids=[0],
        bbox=[0.0, 0.45, 1.0, 0.88],
        observed_prompt_text="蜻蜓",
        observed_raw_text="qin tin\n蜻蜓",
        confidence=0.7,
    )
    values = build_question_values(
        _item(),
        index=1,
        confidence_threshold=0.85,
        localization=localization,
        localization_threshold=0.85,
        localization_max_area_ratio=0.35,
        marks={0: _mark()},
        normalized_tags=["拼音"],
    )

    assert values["crop_region"] == {
        "bbox_source": "unverified",
        "localization_status": "needs_review",
        "index": 1,
    }
    assert values["status"] == "needs_review"


def test_low_confidence_item_requires_review():
    from app.services.vision_recognition import build_question_values

    values = build_question_values(
        _item(confidence=0.7),
        index=1,
        confidence_threshold=0.85,
        localization=None,
        localization_threshold=0.85,
        localization_max_area_ratio=0.35,
        marks={0: _mark()},
        normalized_tags=["拼音"],
    )

    assert values["status"] == "needs_review"


def test_uncertain_segments_require_review_even_with_high_confidence():
    from app.services.vision_recognition import build_question_values

    values = build_question_values(
        _item(confidence=0.99, uncertain_segments=["第一个拼音末尾"]),
        index=0,
        confidence_threshold=0.85,
        localization=None,
        localization_threshold=0.85,
        localization_max_area_ratio=0.35,
        marks={0: _mark()},
        normalized_tags=["拼音"],
    )

    assert values["status"] == "needs_review"


def test_unmatched_localization_discards_bbox_even_with_high_confidence():
    from app.services.vision_recognition import build_question_values

    localization = LocalizationItem(
        index=0,
        matched=False,
        mark_ids=[],
        bbox=None,
        observed_prompt_text=None,
        observed_raw_text=None,
        confidence=0.99,
    )
    values = build_question_values(
        _item(),
        index=0,
        confidence_threshold=0.85,
        localization=localization,
        localization_threshold=0.85,
        localization_max_area_ratio=0.35,
        marks={0: _mark()},
        normalized_tags=["拼音"],
    )

    assert "bbox" not in values["crop_region"]
    assert values["status"] == "needs_review"


def test_localization_missing_assigned_mark_discards_bbox():
    from app.services.vision_recognition import build_question_values

    localization = LocalizationItem(
        index=0,
        matched=True,
        mark_ids=[],
        bbox=[0.7, 0.7, 0.9, 0.9],
        observed_prompt_text="蜻蜓",
        observed_raw_text="qin tin\n蜻蜓",
        confidence=0.99,
    )
    values = build_question_values(
        _item(),
        index=0,
        confidence_threshold=0.85,
        localization=localization,
        localization_threshold=0.85,
        localization_max_area_ratio=0.35,
        marks={0: _mark()},
        normalized_tags=["拼音"],
    )

    assert "bbox" not in values["crop_region"]
    assert values["status"] == "needs_review"


def test_localization_with_mismatched_observed_content_discards_bbox():
    from app.services.vision_recognition import build_question_values

    localization = LocalizationItem(
        index=0,
        matched=True,
        mark_ids=[0],
        bbox=[0.1, 0.2, 0.4, 0.5],
        observed_prompt_text="算式",
        observed_raw_text="suàn shì",
        confidence=0.95,
    )
    values = build_question_values(
        _item(),
        index=0,
        confidence_threshold=0.85,
        localization=localization,
        localization_threshold=0.85,
        localization_max_area_ratio=0.35,
        marks={0: _mark()},
        normalized_tags=["拼音"],
    )

    assert "bbox" not in values["crop_region"]
    assert values["status"] == "needs_review"


def test_image_status_requires_review_when_any_question_does():
    from app.services.vision_recognition import image_status_for

    assert image_status_for([{"status": "confirmed"}, {"status": "needs_review"}]) == "needs_review"
    assert image_status_for([{"status": "confirmed"}]) == "confirmed"


def test_task_claims_image_before_remote_call_and_resets_failed_claim():
    from pathlib import Path

    source = (Path(__file__).parents[2] / "app" / "tasks" / "process_image.py").read_text(encoding="utf-8")

    claim = source.index("with_for_update()")
    remote_call = source.index("recognize_question_batch(")
    assert claim < remote_call
    assert 'image.status != "pending"' in source
    assert 'image.status = "segmented"' in source
    assert 'claimed_image.status = "pending"' in source


def test_task_logs_mark_validation_diagnostics(caplog):
    import logging

    from app.tasks.process_image import log_mark_validation_diagnostics

    question_values = [
        {
            "ocr_raw_json": {
                "error_mark_validation": [
                    {
                        "mark_id": 1,
                        "red_pixel_ratio": 0.004,
                        "red_pixel_min_ratio": 0.005,
                        "accepted": False,
                        "reason": "insufficient_red_pixels",
                    }
                ]
            }
        }
    ]

    with caplog.at_level(logging.INFO):
        log_mark_validation_diagnostics("image-123", question_values)

    assert "image_id=image-123" in caplog.text
    assert '"mark_id":1' in caplog.text
    assert '"red_pixel_ratio":0.004' in caplog.text
    assert '"reason":"insufficient_red_pixels"' in caplog.text
