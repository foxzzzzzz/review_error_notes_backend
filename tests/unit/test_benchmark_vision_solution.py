import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw

from app.services.vision_recognition import (
    ErrorMark,
    LocalizationItem,
    LocalizationResult,
    VisionItem,
    VisionResult,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_vision_solution.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_vision_solution", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_truth(tmp_path, labels):
    path = tmp_path / "truth-regions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pages": {
                    label: {
                        "regions": [
                            {
                                "truth_id": f"{label}-T1",
                                "source_bbox_normalized": [0.1, 0.1, 0.4, 0.4],
                            }
                        ]
                    }
                    for label in labels
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loads_six_dynamic_page_labels_without_hard_coding(tmp_path):
    benchmark = _load_module()
    labels = ["page33", "page34", "page35", "page5", "page7", "page20"]

    loaded = benchmark.load_truth_regions(_write_truth(tmp_path, labels), labels)

    assert list(loaded) == labels
    assert loaded["page20"][0]["truth_id"] == "page20-T1"


def test_truth_audit_reports_missed_false_and_duplicate_regions():
    benchmark = _load_module()
    truth = [
        {"truth_id": "T1", "source_bbox_normalized": [0.1, 0.1, 0.4, 0.4]},
        {"truth_id": "T2", "source_bbox_normalized": [0.6, 0.6, 0.9, 0.9]},
    ]
    predictions = [
        {"prediction_id": "P1", "bbox": [0.1, 0.1, 0.4, 0.4]},
        {"prediction_id": "P2", "bbox": [0.12, 0.12, 0.38, 0.38]},
        {"prediction_id": "P3", "bbox": [0.42, 0.42, 0.52, 0.52]},
    ]

    audit = benchmark.compare_predictions_to_truth(predictions, truth, min_iou=0.2)

    assert audit["missed_truth_ids"] == ["T2"]
    assert audit["false_prediction_ids"] == ["P3"]
    assert audit["duplicate_truth_assignments"] == [
        {"truth_id": "T1", "prediction_ids": ["P1", "P2"]}
    ]
    assert audit["truth_recall"] == 0.5


class FakeOldClient:
    def __init__(self):
        self.calls = []
        self.benchmark_request_records = []

    def recognize(self, image_path, subject_hint=None):
        self.calls.append(("recognize", image_path, subject_hint))
        self.benchmark_request_records.append({"operation": "recognition"})
        return VisionResult(
            items=[
                VisionItem(
                    raw_text="ke wen",
                    instruction="看词语写拼音",
                    prompt_text="课文",
                    normalized_text="ke wen",
                    answer="ke wen",
                    subject="chinese",
                    question_type="write_pinyin",
                    tags=[],
                    difficulty=2,
                    confidence=0.95,
                    uncertain_segments=[],
                )
            ],
            error_marks=[
                ErrorMark(
                    mark_id=0,
                    mark_type="cross",
                    bbox=[0.15, 0.2, 0.25, 0.3],
                    confidence=0.96,
                )
            ],
            ignored_text=[],
        )

    def localize(self, image_path, items, error_marks):
        self.calls.append(("localize", image_path, len(items), len(error_marks)))
        self.benchmark_request_records.append({"operation": "localization"})
        return LocalizationResult(
            items=[
                LocalizationItem(
                    index=0,
                    matched=True,
                    mark_ids=[0],
                    bbox=[0.1, 0.2, 0.3, 0.4],
                    observed_prompt_text="课文",
                    observed_raw_text="ke wen",
                    confidence=0.94,
                )
            ]
        )


class RejectedMarkClient(FakeOldClient):
    def localize(self, image_path, items, error_marks):
        raise AssertionError("localization must not run when every detected mark is rejected")


def test_old_solution_runs_recognition_filter_and_localization_once(tmp_path):
    benchmark = _load_module()
    image_path = tmp_path / "page.jpg"
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 35, 55, 65), fill=(220, 20, 20))
    image.save(image_path)
    client = FakeOldClient()

    result = benchmark.run_old_solution_page(
        image_path=image_path,
        subject="chinese",
        client=client,
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.005,
        red_pixel_expansion_ratio=0.08,
    )

    assert result["solution_id"] == "old_solution"
    assert result["llm_request_count"] == 2
    assert [item["bbox"] for item in result["predictions"]] == [
        [0.1, 0.2, 0.3, 0.4]
    ]
    assert result["timing_ms"]["ocr"] == 0
    assert [call[0] for call in client.calls] == ["recognize", "localize"]


def test_old_solution_skips_localization_when_all_detected_marks_are_rejected(
    tmp_path,
):
    benchmark = _load_module()
    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (200, 200), "white").save(image_path)
    client = RejectedMarkClient()

    result = benchmark.run_old_solution_page(
        image_path=image_path,
        subject="chinese",
        client=client,
        mark_confidence_threshold=0.85,
        red_pixel_min_ratio=0.005,
        red_pixel_expansion_ratio=0.08,
    )

    assert result["predictions"] == []
    assert result["llm_request_count"] == 1
    assert [call[0] for call in client.calls] == ["recognize"]
