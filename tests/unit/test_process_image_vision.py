from app.services.vision_recognition import VisionItem


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
        "normalized_text": "qīng tíng\n蜻蜓",
        "answer": "qīng tíng",
        "subject": "chinese",
        "question_type": "pinyin",
        "tags": ["拼音"],
        "difficulty": 2,
        "confidence": 0.92,
        "uncertain_segments": [],
        "bbox": [0.1, 0.2, 0.4, 0.4],
    }
    data.update(overrides)
    return VisionItem(**data)


def test_question_values_preserve_raw_writing_and_normalized_content():
    from app.services.vision_recognition import build_question_values

    values = build_question_values(_item(), index=0, confidence_threshold=0.85)

    assert values["ocr_text"] == "qin tin\n蜻蜓"
    assert values["ocr_text"] != values["ocr_raw_json"]["normalized_text"]
    assert values["ocr_answer"] == "qīng tíng"
    assert values["ocr_raw_json"]["provider"] == "minimax"
    assert values["ocr_raw_json"]["confidence"] == 0.92
    assert values["ocr_raw_json"]["raw_text"] == values["ocr_text"]
    assert values["ocr_raw_json"]["answer"] == values["ocr_answer"]
    assert values["ocr_raw_json"]["subject"] == "chinese"
    assert values["ocr_raw_json"]["bbox"] == [0.1, 0.2, 0.4, 0.4]
    assert values["crop_region"] == {
        "bbox": [0.1, 0.2, 0.4, 0.4],
        "bbox_format": "normalized_ltrb",
        "index": 0,
    }
    assert values["status"] == "confirmed"


def test_question_values_label_minimax_corner_bbox_format():
    from app.services.vision_recognition import build_question_values

    bbox = [0.0, 0.45, 1.0, 0.88]
    values = build_question_values(_item(bbox=bbox), index=1, confidence_threshold=0.85)

    assert values["crop_region"] == {
        "bbox": bbox,
        "bbox_format": "normalized_ltrb",
        "index": 1,
    }


def test_low_confidence_item_requires_review():
    from app.services.vision_recognition import build_question_values

    values = build_question_values(_item(confidence=0.7), index=1, confidence_threshold=0.85)

    assert values["status"] == "needs_review"


def test_uncertain_segments_require_review_even_with_high_confidence():
    from app.services.vision_recognition import build_question_values

    values = build_question_values(
        _item(confidence=0.99, uncertain_segments=["第一个拼音末尾"]),
        index=0,
        confidence_threshold=0.85,
    )

    assert values["status"] == "needs_review"


def test_image_status_requires_review_when_any_question_does():
    from app.services.vision_recognition import image_status_for

    assert image_status_for([{"status": "confirmed"}, {"status": "needs_review"}]) == "needs_review"
    assert image_status_for([{"status": "confirmed"}]) == "confirmed"


def test_task_claims_image_before_remote_call_and_resets_failed_claim():
    from pathlib import Path

    source = (Path(__file__).parents[2] / "app" / "tasks" / "process_image.py").read_text(encoding="utf-8")

    claim = source.index("with_for_update()")
    remote_call = source.index(".recognize(")
    assert claim < remote_call
    assert 'image.status != "pending"' in source
    assert 'image.status = "segmented"' in source
    assert 'claimed_image.status = "pending"' in source
