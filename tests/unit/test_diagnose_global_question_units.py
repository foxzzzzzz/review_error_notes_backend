import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "diagnose_global_question_units.py"


def _load_diagnostic():
    scripts_path = str(BACKEND_ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "diagnose_global_question_units", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path):
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    red = (0, 0, 180)
    for y in (60, 240):
        cv2.line(image, (80, y), (720, y), red, 5)
    for x in (80, 720):
        cv2.line(image, (x, 60), (x, 240), red, 5)
    image_path = tmp_path / "page33.jpg"
    assert cv2.imwrite(str(image_path), image)
    anchor_dir = tmp_path / "anchors/page33/cross-anchor-experiment"
    anchor_dir.mkdir(parents=True)
    (anchor_dir / "confirmed-crosses.json").write_text(
        json.dumps([{"cross_id": 1, "bbox": [0.40, 0.20, 0.42, 0.22]}]),
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth-regions.json"
    truth_path.write_text(
        json.dumps(
            {
                "pages": {
                    "page33": {
                        "regions": [
                            {
                                "truth_id": "T1",
                                "source_bbox_normalized": [0.1, 0.1, 0.9, 0.4],
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return image_path, tmp_path / "anchors", truth_path


def _install_fake_ocr(monkeypatch, diagnostic, lines):
    class FakeVerifier:
        def recognize_page(self, image_path, max_edge):
            del image_path, max_edge
            return SimpleNamespace(
                status="available",
                error_code=None,
                lines=[
                    SimpleNamespace(model_dump=lambda mode, line=line: dict(line))
                    for line in lines
                ],
            )

    monkeypatch.setattr(diagnostic, "_ocr_verifier", lambda: FakeVerifier())


def _arguments(image_path, anchors_root, truth_path, output):
    return [
        f"page33={image_path}",
        "--anchors-root",
        str(anchors_root),
        "--truth-regions",
        str(truth_path),
        "--output",
        str(output),
    ]


def test_script_can_start_directly_from_backend_root():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--anchors-root" in result.stdout


def test_cli_writes_local_artifacts_and_zero_llm_requests(tmp_path, monkeypatch):
    diagnostic = _load_diagnostic()
    image_path, anchors_root, truth_path = _fixture(tmp_path)
    _install_fake_ocr(
        monkeypatch,
        diagnostic,
        [{"text": "answer", "confidence": 0.95, "bbox": [0.2, 0.16, 0.6, 0.22]}],
    )
    output = tmp_path / "out"

    exit_code = diagnostic.main(
        _arguments(image_path, anchors_root, truth_path, output)
    )

    assert exit_code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["llm_request_count"] == 0
    assert (output / "page33/global-question-units.json").is_file()
    assert (output / "page33/anchor-unit-candidates.json").is_file()
    assert (output / "page33/question-unit-oracle-audit.json").is_file()
    assert (output / "page33/global-question-units-overlay.jpg").is_file()
    assert (output / "page33/anchor-unit-candidates-overlay.jpg").is_file()


def test_cli_repeat_outputs_have_identical_geometry(tmp_path, monkeypatch):
    diagnostic = _load_diagnostic()
    image_path, anchors_root, truth_path = _fixture(tmp_path)
    _install_fake_ocr(
        monkeypatch,
        diagnostic,
        [{"text": "answer", "confidence": 0.95, "bbox": [0.2, 0.16, 0.6, 0.22]}],
    )

    assert diagnostic.main(
        _arguments(image_path, anchors_root, truth_path, tmp_path / "first")
    ) == 0
    assert diagnostic.main(
        _arguments(image_path, anchors_root, truth_path, tmp_path / "second")
    ) == 0
    first = json.loads((tmp_path / "first/summary.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "second/summary.json").read_text(encoding="utf-8"))
    assert first["page_summaries"][0]["geometry_fingerprint"] == second[
        "page_summaries"
    ][0]["geometry_fingerprint"]


def test_cli_keeps_running_when_ocr_returns_no_lines(tmp_path, monkeypatch):
    diagnostic = _load_diagnostic()
    image_path, anchors_root, truth_path = _fixture(tmp_path)
    _install_fake_ocr(monkeypatch, diagnostic, [])
    output = tmp_path / "out"

    assert diagnostic.main(
        _arguments(image_path, anchors_root, truth_path, output)
    ) == 0
    payload = json.loads(
        (output / "page33/global-question-units.json").read_text(encoding="utf-8")
    )
    assert payload["units"]
    assert all("ocr_missing" in item["risk_flags"] for item in payload["units"])


def _semantic_mapping():
    return {
        "anchor_candidates": {
            "1": [
                {"question_unit_id": "U1", "rank": 1},
                {"question_unit_id": "U2", "rank": 2},
            ],
            "2": [{"question_unit_id": "U3", "rank": 1}],
            "3": [],
        },
        "unassigned_anchors": [
            {"cross_id": 3, "reason": "no_unit_within_distance"}
        ],
    }


def _decision(cross_id, *, selected=None, validity="valid", status="incorrect"):
    return {
        "cross_id": cross_id,
        "anchor_validity": validity,
        "selected_unit_id": selected,
        "question_status": status,
        "boundary_fit": "complete",
        "evidence": ["red_cross"],
        "confidence": 0.9,
    }


def test_semantic_prompt_uses_model_as_judge_without_allowing_geometry():
    diagnostic = _load_diagnostic()

    assert "锚点" in diagnostic.SEMANTIC_JUDGE_PROMPT
    assert "实际答题" in diagnostic.SEMANTIC_JUDGE_PROMPT
    assert "红圈" in diagnostic.SEMANTIC_JUDGE_PROMPT
    assert "不得返回" in diagnostic.SEMANTIC_JUDGE_PROMPT
    assert "bbox" in diagnostic.SEMANTIC_JUDGE_PROMPT
    assert "supplemental_wrong_unit_ids" in diagnostic.SEMANTIC_JUDGE_PROMPT


def test_semantic_audit_rejects_duplicate_cross_and_foreign_unit_ids():
    diagnostic = _load_diagnostic()
    decisions = [
        _decision(1, selected="U1"),
        _decision(1, selected="U2"),
        _decision(2, selected="U1"),
    ]

    audit = diagnostic.audit_semantic_judgment(
        decisions, ["UNKNOWN"], _semantic_mapping()
    )

    assert audit["accepted"] == []
    assert {item["reason"] for item in audit["violations"]} == {
        "duplicate_anchor",
        "unit_not_allowed_for_anchor",
        "unknown_supplemental_unit",
        "missing_anchor",
    }


def test_semantic_unit_sets_keep_strict_selection_and_safe_rank_one_fallback():
    diagnostic = _load_diagnostic()
    audit = {
        "accepted": [
            _decision(1, selected="U2"),
            _decision(
                2,
                selected=None,
                validity="uncertain",
                status="uncertain",
            ),
        ],
        "accepted_supplemental_unit_ids": [],
        "violations": [],
    }

    result = diagnostic.build_semantic_unit_sets(audit, _semantic_mapping())

    assert result["strict_unit_ids"] == ["U2"]
    assert result["recall_safe_unit_ids"] == ["U2", "U3"]
    assert result["needs_review"] == [
        {"cross_id": 2, "fallback_unit_id": "U3", "reason": "semantic_uncertain"},
        {"cross_id": 3, "fallback_unit_id": None, "reason": "unassigned_anchor"},
    ]


def test_cli_semantic_judge_makes_exactly_one_request(tmp_path, monkeypatch):
    diagnostic = _load_diagnostic()
    image_path, anchors_root, truth_path = _fixture(tmp_path)
    _install_fake_ocr(
        monkeypatch,
        diagnostic,
        [{"text": "answer", "confidence": 0.95, "bbox": [0.2, 0.16, 0.6, 0.22]}],
    )
    calls = []

    class FakeClient:
        max_edge = 2048
        jpeg_quality = 90
        diagnostic_event_sink = None

        @classmethod
        def from_settings(cls):
            return cls()

        def _request(self, payload, result_model, diagnostic_context):
            calls.append((payload, diagnostic_context))
            return result_model.model_validate(
                {
                    "decisions": [
                        {
                            "cross_id": 1,
                            "anchor_validity": "valid",
                            "selected_unit_id": "U-S01-R01-C01",
                            "question_status": "incorrect",
                            "boundary_fit": "complete",
                            "evidence": ["red_cross", "answer_mismatch"],
                            "confidence": 0.92,
                        }
                    ],
                    "supplemental_wrong_unit_ids": [],
                }
            )

    monkeypatch.setattr(diagnostic, "MiniMaxVisionClient", FakeClient)
    monkeypatch.setattr(
        diagnostic,
        "prepare_image_data_url",
        lambda image_path, max_edge, jpeg_quality, context: "data:image/jpeg;base64,fake",
    )
    output = tmp_path / "semantic"
    arguments = _arguments(image_path, anchors_root, truth_path, output)
    arguments.extend(["--run-semantic-judge", "--subject", "chinese"])

    assert diagnostic.main(arguments) == 0

    assert len(calls) == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["llm_request_count"] == 1
    page_dir = output / "page33"
    assert (page_dir / "semantic-judge-montage.jpg").is_file()
    assert (page_dir / "semantic-judge-response.json").is_file()
    assert (page_dir / "semantic-judge-audit.json").is_file()
    assert (page_dir / "semantic-strict-comparison.json").is_file()
    assert (page_dir / "semantic-recall-safe-comparison.json").is_file()
