import json
from pathlib import Path

from PIL import Image


def test_renderer_writes_stage_overlays_and_inflation_summary(tmp_path):
    from scripts.chinese_marked_evidence_stage_audit import render_stage_audit_report

    image_path = tmp_path / "mounted-source"
    Image.new("RGB", (200, 120), "white").save(image_path, format="JPEG")
    review_path = tmp_path / "review-images.json"
    review_path.write_text(
        json.dumps(
            [
                {
                    "image_id": "image-p003",
                    "question_count": 2,
                    "questions": [
                        {
                            "id": "question-1",
                            "ocr_raw_json": {
                                "stage_audit": {
                                    "red_scan_ms": 5.5,
                                    "red_components": [
                                        {
                                            "component_id": 0,
                                            "bbox": [0.1, 0.1, 0.2, 0.2],
                                            "pixel_count": 20,
                                        }
                                    ],
                                    "evidence_groups": [
                                        {
                                            "region_id": 0,
                                            "bbox": [0.08, 0.08, 0.22, 0.22],
                                            "member_component_ids": [0],
                                        }
                                    ],
                                    "ocr_lines": [
                                        {
                                            "ocr_line_id": 0,
                                            "text": "题面",
                                            "confidence": 0.9,
                                            "bbox": [0.3, 0.1, 0.5, 0.2],
                                        }
                                    ],
                                    "mark_attempts": [
                                        {
                                            "attempt": 1,
                                            "entry": "direct",
                                            "marks": [
                                                {
                                                    "source_ref": "attempt-1:0",
                                                    "mark_type": "circle",
                                                    "bbox": [0.1, 0.1, 0.2, 0.2],
                                                    "accepted": True,
                                                }
                                            ],
                                        }
                                    ],
                                    "merged_primitives": [],
                                    "mark_events": [
                                        {
                                            "mark_id": 0,
                                            "mark_type": "circle",
                                            "bbox": [0.1, 0.1, 0.2, 0.2],
                                        },
                                        {
                                            "mark_id": 1,
                                            "mark_type": "cross",
                                            "bbox": [0.6, 0.1, 0.7, 0.2],
                                        },
                                    ],
                                    "localizations": [
                                        {
                                            "mark_id": 0,
                                            "bbox": [0.05, 0.05, 0.3, 0.4],
                                            "answer_bbox": [0.1, 0.1, 0.2, 0.2],
                                        },
                                        {
                                            "mark_id": 1,
                                            "bbox": [0.5, 0.05, 0.8, 0.4],
                                            "answer_bbox": [0.6, 0.1, 0.7, 0.2],
                                        },
                                    ],
                                },
                                "local_ocr_page": {
                                    "status": "available",
                                    "duration_ms": 12.25,
                                },
                                "three_stage": {
                                    "mark_llm_ms": 100.0,
                                    "localization_llm_ms": 200.0,
                                    "content_llm_ms": 300.0,
                                    "content_item_count": 2,
                                },
                                "evidence_timing": {
                                    "before_persistence": {"elapsed_ms": 650.0},
                                    "pre_commit": {"elapsed_ms": 675.0},
                                },
                                "evidence_bundle": {
                                    "identity": {
                                        "mark_id": 0,
                                        "question_geometry": {
                                            "question_bbox": [0.05, 0.05, 0.3, 0.4]
                                        },
                                    },
                                    "fields": {},
                                },
                            },
                        },
                        {
                            "id": "question-2",
                            "ocr_raw_json": {
                                "evidence_bundle": {
                                    "identity": {
                                        "mark_id": 1,
                                        "question_geometry": {
                                            "question_bbox": [0.5, 0.05, 0.8, 0.4]
                                        },
                                    },
                                    "fields": {},
                                }
                            },
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"

    summary = render_stage_audit_report(
        review_path=review_path,
        output_dir=output_dir,
        pages=[
            {
                "label": "P003",
                "image_id": "image-p003",
                "image_path": image_path,
                "source_name": "P003.jpg",
                "truth_count": 1,
            }
        ],
    )

    page_dir = output_dir / "P003"
    assert (page_dir / "00-source.jpg").is_file()
    assert (page_dir / "01-red-components.png").is_file()
    assert (page_dir / "03-ocr-lines.png").is_file()
    assert (page_dir / "04-mark-attempt-1.png").is_file()
    assert (page_dir / "06-merged-events.png").is_file()
    assert (page_dir / "07-question-localizations.png").is_file()
    assert (page_dir / "09-persisted-candidates.png").is_file()
    assert (page_dir / "audit.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "summary.md").is_file()
    assert summary[0]["truth_count"] == 1
    assert summary[0]["mark_event_count"] == 2
    assert summary[0]["persisted_candidate_count"] == 2
    assert summary[0]["event_inflation"] == 1
    assert summary[0]["red_scan_ms"] == 5.5
    assert summary[0]["full_page_ocr_ms"] == 12.25
    assert summary[0]["mark_detection_ms"] == 100.0
    assert summary[0]["question_localization_ms"] == 200.0
    assert summary[0]["content_recognition_ms"] == 300.0
    assert summary[0]["before_persistence_elapsed_ms"] == 650.0
    assert summary[0]["pre_commit_elapsed_ms"] == 675.0
