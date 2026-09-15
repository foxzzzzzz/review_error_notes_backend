import pytest

from app.services.vision_recognition import ErrorMark, LocalizationItem, VisionItem


@pytest.mark.parametrize(
    ("enabled", "subjects", "subject", "correction", "expected"),
    [
        (False, "chinese", "chinese", None, False),
        (True, "chinese", "math", None, False),
        (True, "chinese", "english", None, False),
        (True, "chinese", None, None, False),
        (True, "", "chinese", None, False),
        (True, " Chinese , math ", "CHINESE", None, True),
        (True, "chinese", "chinese", "force_unmarked", False),
    ],
)
def test_chinese_marked_evidence_request_gate(
    monkeypatch, enabled, subjects, subject, correction, expected
):
    from app.tasks import process_image as process_image_module

    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_ENABLED",
        enabled,
    )
    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_SUBJECTS",
        subjects,
    )

    assert process_image_module.evidence_mode_requested_for(
        subject,
        correction,
    ) is expected


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


def test_display_bbox_expands_and_moves_toward_assigned_mark():
    from app.services.vision_recognition import marker_focused_display_bbox

    mark = ErrorMark(
        mark_id=0,
        mark_type="circle",
        bbox=[0.55, 0.42, 0.59, 0.46],
        confidence=0.96,
    )

    display_bbox = marker_focused_display_bbox(
        localization_bbox=[0.4, 0.4, 0.6, 0.6],
        mark_ids=[0],
        marks={0: mark},
        padding_ratio=0.15,
    )

    assert display_bbox == pytest.approx([0.4, 0.34, 0.66, 0.6])
    assert display_bbox[0] <= 0.4
    assert display_bbox[1] <= 0.4
    assert display_bbox[2] >= 0.6
    assert display_bbox[3] >= 0.6


def test_display_bbox_stays_in_image_at_edge():
    from app.services.vision_recognition import marker_focused_display_bbox

    mark = ErrorMark(
        mark_id=0,
        mark_type="circle",
        bbox=[0.01, 0.12, 0.05, 0.16],
        confidence=0.96,
    )

    display_bbox = marker_focused_display_bbox(
        localization_bbox=[0.0, 0.1, 0.2, 0.3],
        mark_ids=[0],
        marks={0: mark},
        padding_ratio=0.15,
    )

    assert display_bbox == pytest.approx([0.0, 0.04, 0.26, 0.3])
    assert all(0 <= coordinate <= 1 for coordinate in display_bbox)


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
    assert values["review_status"] == "confirmed"


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
    assert values["review_status"] == "needs_review"


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

    assert values["review_status"] == "needs_review"


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

    assert values["review_status"] == "needs_review"


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
    assert values["review_status"] == "needs_review"


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
    assert values["review_status"] == "needs_review"


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
    assert values["review_status"] == "needs_review"


def test_image_status_requires_review_when_any_question_does():
    from app.services.vision_recognition import image_status_for

    assert image_status_for(
        [
            {"review_status": "confirmed"},
            {"review_status": "needs_review"},
        ]
    ) == "needs_review"
    assert image_status_for([{"review_status": "confirmed"}]) == "confirmed"


def test_task_claims_image_before_remote_call_and_marks_failed_claim():
    from pathlib import Path

    source = (Path(__file__).parents[2] / "app" / "tasks" / "process_image.py").read_text(encoding="utf-8")

    claim = source.index("with_for_update()")
    remote_call = source.index("recognize_question_batch(")
    assert claim < remote_call
    assert 'image.status != "pending"' in source
    assert 'image.status = "segmented"' in source
    assert 'claimed_image.status = "failed"' in source
    assert 'claimed_image.error_code' in source


def test_task_rechecks_account_status_after_recognition_before_persistence():
    from pathlib import Path

    source = (
        Path(__file__).parents[2] / "app" / "tasks" / "process_image.py"
    ).read_text(encoding="utf-8")

    remote_call = source.index("recognize_question_batch(")
    persistence = source.index("for values in question_values")
    active_checks = [
        index
        for index in range(len(source))
        if source.startswith('Account.status == "active"', index)
    ]

    assert len(active_checks) >= 2
    assert remote_call < active_checks[-1] < persistence


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


def test_task_logs_safe_localization_counts_without_recognized_text(caplog):
    import logging

    from app.tasks.process_image import log_mark_validation_diagnostics

    question_values = [
        {
            "ocr_text": "学生隐私作答",
            "ocr_raw_json": {
                "error_mark_validation": [],
                "localization_batch_validation": {
                    "status": "validated",
                    "error_code": None,
                    "error_reason": None,
                    "returned_count": 2,
                    "validated_count": 2,
                    "verified_count": 1,
                    "reliable_mark_count": 1,
                },
            },
        }
    ]

    with caplog.at_level(logging.INFO):
        log_mark_validation_diagnostics("image-123", question_values)

    assert "localization_validation image_id=image-123" in caplog.text
    assert '"returned_count":2' in caplog.text
    assert '"reliable_mark_count":1' in caplog.text
    assert "学生隐私作答" not in caplog.text


def test_task_persists_placeholder_evidence_after_deadline_without_result_item(
    monkeypatch,
    caplog,
):
    from types import SimpleNamespace

    from app.tasks import process_image as process_image_module

    image = SimpleNamespace(
        id="image-8",
        student_id="student-8",
        subject="chinese",
        grade=3,
        semester=1,
        status="pending",
        question_count=0,
        recognition_correction=None,
        error_code=None,
        error_message=None,
    )
    added = []
    commits = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def scalar(self, _statement):
            return image

        def add(self, question):
            added.append(question)

        def commit(self):
            commits.append(image.status)

    class FakeEngine:
        def __init__(self):
            self.disposed = False

        def dispose(self):
            self.disposed = True

    engine = FakeEngine()
    actual_client = object()
    actual_ocr = object()
    captured = {}
    persisted_kwargs = []

    def evidence_value(mark_id):
        return {
            "recognition_pipeline": "chinese_marked_evidence_v1",
            "mark_status": "needs_review",
            "question_evidence_status": "insufficient",
            "answer_status": "unresolved",
            "collection_status": "pending_review",
            "review_status": "needs_review",
            "ocr_text": None,
            "ocr_answer": None,
            "subject": "chinese",
            "question_type": None,
            "tags": [],
            "difficulty": None,
            "crop_region": {"mark_ids": [mark_id]},
            "ocr_raw_json": {
                "evidence_bundle": {
                    "schema_version": 1,
                    "identity": {
                        "image_id": "image-8",
                        "mark_id": mark_id,
                        "question_geometry": {"bbox": [0.1, 0.2, 0.3, 0.4]},
                    },
                },
                "three_stage": {"recognition_pipeline": "three_stage"},
                "local_ocr_page": {"status": "disabled", "duration_ms": 0.0},
            },
        }

    values = [evidence_value(0), evidence_value(1)]

    def fake_recognize_question_batch(**kwargs):
        captured.update(kwargs)
        return (
            SimpleNamespace(
                items=[],
                ignored_text=[],
                recognition_pipeline="chinese_marked_evidence_v1",
            ),
            values,
        )

    ticks = iter([100.0, 102.0, 108.0, 110.0])
    monkeypatch.setattr(process_image_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_SUBJECTS",
        "chinese",
    )
    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_PAGE_TIMEOUT_SECONDS",
        3.0,
    )
    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_OCR_CROP_RECHECK_LIMIT",
        1,
    )
    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_LOCALIZATION_RECHECK_LIMIT",
        2,
    )
    monkeypatch.setattr(
        process_image_module.settings,
        "LOCAL_OCR_CROP_RECHECK_LIMIT",
        7,
    )
    monkeypatch.setattr(process_image_module, "create_engine", lambda _url: engine)
    monkeypatch.setattr(process_image_module, "Session", lambda _engine: FakeSession())
    monkeypatch.setattr(
        process_image_module,
        "scan_red_mark_regions",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="detected",
            regions=[],
            duration_ms=1.0,
        ),
    )
    monkeypatch.setattr(
        process_image_module,
        "create_vision_client",
        lambda: actual_client,
    )
    monkeypatch.setattr(
        process_image_module,
        "RapidOCRVerifier",
        lambda **_kwargs: actual_ocr,
    )
    monkeypatch.setattr(
        process_image_module,
        "recognize_question_batch",
        fake_recognize_question_batch,
    )

    def fake_wrong_question(**kwargs):
        persisted_kwargs.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(process_image_module, "WrongQuestion", fake_wrong_question)

    import logging

    with caplog.at_level(logging.INFO):
        process_image_module.process_image.run("image-8", "worksheet.jpg")

    assert captured["client"] is actual_client
    assert captured["ocr_verifier"] is actual_ocr
    assert captured["evidence_mode"] is True
    assert captured["image_id"] == "image-8"
    assert captured["deadline"] == 103.0
    assert captured["ocr_crop_recheck_limit"] == 7
    assert captured["evidence_ocr_crop_recheck_limit"] == 1
    assert captured["evidence_localization_recheck_limit"] == 2
    assert len(added) == len(persisted_kwargs) == 2
    assert image.question_count == 2
    assert image.status == "needs_review"
    assert commits == ["segmented", "needs_review"]
    for source, persisted in zip(values, persisted_kwargs):
        assert persisted["recognition_pipeline"] == source["recognition_pipeline"]
        assert persisted["mark_status"] == source["mark_status"]
        assert persisted["question_evidence_status"] == source["question_evidence_status"]
        assert persisted["answer_status"] == source["answer_status"]
        assert persisted["collection_status"] == source["collection_status"]
        assert persisted["ocr_raw_json"]["evidence_bundle"] == source["ocr_raw_json"]["evidence_bundle"]
        timing = persisted["ocr_raw_json"]["evidence_timing"]
        assert timing == {
            "before_persistence": {
                "timeout_seconds": 3.0,
                "elapsed_seconds": 2.0,
                "remaining_seconds": 1.0,
                "exhausted": False,
            },
            "pre_commit": {
                "timeout_seconds": 3.0,
                "elapsed_seconds": 8.0,
                "remaining_seconds": 0.0,
                "exhausted": True,
            },
        }
    assert '"elapsed_seconds":10.0' in caplog.text
    assert '"remaining_seconds":0.0' in caplog.text
    assert '"exhausted":true' in caplog.text
    assert engine.disposed is True


def test_process_actual_unmarked_is_identical_when_evidence_is_requested(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from PIL import Image

    from app.services.local_ocr_verification import OCRPageEvidence, OCRVerification
    from app.services.vision_recognition import (
        LocalizationItem,
        LocalizationResult,
        VisionItem,
        VisionResult,
    )
    from app.tasks import process_image as process_image_module

    image_path = tmp_path / "white-page.jpg"
    Image.new("RGB", (600, 300), "white").save(image_path)

    class UnmarkedClient:
        def __init__(self):
            self.recognize_calls = 0
            self.localize_calls = 0

        def recognize(
            self,
            image_path,
            subject_hint=None,
            recognition_correction=None,
            recognition_mode="marked",
            local_red_regions=None,
        ):
            self.recognize_calls += 1
            return VisionResult(
                items=[
                    VisionItem(
                        raw_text=f"学生答案{index}",
                        instruction="填写答案",
                        prompt_text=f"题目{index}",
                        normalized_text=None,
                        answer=f"参考答案{index}",
                        subject="chinese",
                        question_type="fill_blank",
                        tags=[],
                        difficulty=2,
                        confidence=0.99 - index / 100,
                        uncertain_segments=[],
                    )
                    for index in range(3)
                ],
                error_marks=[],
                ignored_text=[],
            )

        def localize(self, image_path, items, error_marks):
            self.localize_calls += 1
            assert error_marks == []
            return LocalizationResult(
                items=[
                    LocalizationItem(
                        index=index,
                        matched=True,
                        mark_ids=[],
                        bbox=[
                            0.05,
                            0.05 + index * 0.3,
                            0.95,
                            0.25 + index * 0.3,
                        ],
                        observed_prompt_text=item.prompt_text,
                        observed_raw_text=item.raw_text,
                        confidence=0.95,
                    )
                    for index, item in enumerate(items)
                ]
            )

    class CountingOCR:
        enabled = True
        line_confidence_threshold = 0.85
        min_effective_characters = 2
        support_similarity_threshold = 0.8
        contradiction_similarity_threshold = 0.9

        def __init__(self):
            self.page_calls = 0
            self.crop_calls = 0

        def recognize_page(self, _image_path, _max_edge):
            self.page_calls += 1
            return OCRPageEvidence(status="available", lines=[])

        def verify_crop(self, *_args, **_kwargs):
            self.crop_calls += 1
            return OCRVerification(status="inconclusive")

    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_SUBJECTS",
        "chinese",
    )
    monkeypatch.setattr(
        process_image_module.settings,
        "LOCAL_OCR_CROP_RECHECK_LIMIT",
        2,
    )
    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_OCR_CROP_RECHECK_LIMIT",
        0,
    )

    def run_flag(enabled):
        claimed_image = SimpleNamespace(
            id=f"image-unmarked-{enabled}",
            student_id="student-8",
            subject="chinese",
            grade=3,
            semester=1,
            status="pending",
            question_count=0,
            recognition_correction=None,
            error_code=None,
            error_message=None,
        )
        persisted = []
        client = UnmarkedClient()
        verifier = CountingOCR()

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def scalar(self, _statement):
                return claimed_image

            def add(self, question):
                persisted.append(question)

            def commit(self):
                pass

        class FakeEngine:
            def dispose(self):
                pass

        monkeypatch.setattr(
            process_image_module.settings,
            "CHINESE_MARKED_EVIDENCE_ENABLED",
            enabled,
        )
        monkeypatch.setattr(
            process_image_module,
            "create_engine",
            lambda _url: FakeEngine(),
        )
        monkeypatch.setattr(
            process_image_module,
            "Session",
            lambda _engine: FakeSession(),
        )
        monkeypatch.setattr(
            process_image_module,
            "create_vision_client",
            lambda: client,
        )
        monkeypatch.setattr(
            process_image_module,
            "RapidOCRVerifier",
            lambda **_kwargs: verifier,
        )
        monkeypatch.setattr(
            process_image_module,
            "WrongQuestion",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )

        process_image_module.process_image.run(
            claimed_image.id,
            str(image_path),
        )
        return {
            "page_calls": verifier.page_calls,
            "crop_calls": verifier.crop_calls,
            "statuses": [question.collection_status for question in persisted],
            "persisted_count": len(persisted),
            "question_count": claimed_image.question_count,
            "image_status": claimed_image.status,
            "recognize_calls": client.recognize_calls,
            "localize_calls": client.localize_calls,
        }

    disabled = run_flag(False)
    requested_but_unmarked = run_flag(True)

    assert disabled == requested_but_unmarked == {
        "page_calls": 1,
        "crop_calls": 2,
        "statuses": ["pending_review", "pending_review", "pending_review"],
        "persisted_count": 3,
        "question_count": 3,
        "image_status": "needs_review",
        "recognize_calls": 1,
        "localize_calls": 1,
    }


def test_process_evidence_neutralizes_real_rapidocr_none_score_and_persists(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from PIL import Image

    from app.services.local_ocr_verification import RapidOCRVerifier
    from app.services import vision_recognition
    from app.tasks import process_image as process_image_module

    image_path = tmp_path / "marked-page.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    image = SimpleNamespace(
        id="image-none-score",
        student_id="student-8",
        subject="chinese",
        grade=3,
        semester=1,
        status="pending",
        question_count=0,
        recognition_correction=None,
        error_code=None,
        error_message=None,
    )
    persisted = []
    captured = {}

    class Engine:
        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(
                txts=["OCR line"],
                scores=[None],
                boxes=None,
            )

    verifier = RapidOCRVerifier(
        enabled=True,
        library_version="3.9.1",
        engine_name="rapidocr",
        model_version="PP-OCRv5",
        model_type="mobile",
        model_path=None,
        max_pixels=40_000_000,
        line_confidence_threshold=0.85,
        min_effective_characters=2,
        support_similarity_threshold=0.8,
        contradiction_similarity_threshold=0.9,
        engine_factory=Engine,
    )

    def fake_three_stage(**kwargs):
        captured.update(kwargs)
        return [
            {
                "recognition_pipeline": "chinese_marked_evidence_v1",
                "mark_status": "needs_review",
                "question_evidence_status": "insufficient",
                "answer_status": "unresolved",
                "collection_status": "pending_review",
                "review_status": "needs_review",
                "ocr_text": None,
                "ocr_answer": None,
                "subject": "chinese",
                "question_type": None,
                "tags": [],
                "difficulty": None,
                "crop_region": {"mark_ids": [0]},
                "ocr_raw_json": {
                    "evidence_bundle": {
                        "schema_version": 1,
                        "identity": {
                            "image_id": image.id,
                            "mark_id": 0,
                            "question_geometry": {
                                "bbox": [0.1, 0.2, 0.3, 0.4]
                            },
                        },
                    }
                },
            }
        ], {}, [], {"recognition_pipeline": "three_stage"}

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def scalar(self, _statement):
            return image

        def add(self, question):
            persisted.append(question)

        def commit(self):
            pass

    class FakeEngine:
        def dispose(self):
            pass

    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        process_image_module.settings,
        "CHINESE_MARKED_EVIDENCE_SUBJECTS",
        "chinese",
    )
    monkeypatch.setattr(
        process_image_module,
        "scan_red_mark_regions",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="detected",
            regions=[],
            duration_ms=0.0,
        ),
    )
    monkeypatch.setattr(process_image_module, "create_engine", lambda _url: FakeEngine())
    monkeypatch.setattr(process_image_module, "Session", lambda _engine: FakeSession())
    monkeypatch.setattr(process_image_module, "create_vision_client", object)
    monkeypatch.setattr(process_image_module, "RapidOCRVerifier", lambda **_kwargs: verifier)
    monkeypatch.setattr(vision_recognition, "recognize_marked_three_stage", fake_three_stage)
    monkeypatch.setattr(
        process_image_module,
        "WrongQuestion",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    process_image_module.process_image.run(image.id, str(image_path))

    assert captured["ocr_page_evidence"].status == "unavailable"
    assert captured["ocr_page_evidence"].error_code == "page_ocr_invalid"
    assert len(persisted) == image.question_count == 1
    assert persisted[0].collection_status == "pending_review"
    assert image.status == "needs_review"
