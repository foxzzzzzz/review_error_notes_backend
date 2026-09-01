import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "replay_cross_anchor_rescue.py"
CONFIG_PATH = BACKEND_ROOT / "scripts" / "cv_cross_experiment_config.json"


def _load_replay_module():
    spec = importlib.util.spec_from_file_location(
        "replay_cross_anchor_rescue", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_replay_directory_writes_rescue_and_truth_artifacts(tmp_path):
    replay = _load_replay_module()
    archive = tmp_path / "run-a"
    page = archive / "pages" / "sample"
    cross = page / "cross-anchor-experiment"
    cv = page / "cv-cross-experiment"
    source = page / "source.png"
    source.parent.mkdir(parents=True)
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 39, 39), fill=(220, 30, 30))
    image.save(source)
    _write_json(cv / "candidates.json", [])
    _write_json(cross / "confirmed-crosses.json", [])
    _write_json(
        cross / "llm1-independent-scan.json",
        {"crosses": [{"bbox": [0.2, 0.2, 0.4, 0.4], "confidence": 0.95}]},
    )
    _write_json(
        cross / "llm1-fallback-candidate-verification.json",
        {
            "verdicts": [
                {
                    "candidate_id": 0,
                    "disposition": "rejected",
                    "confidence": 0.8,
                }
            ]
        },
    )
    truth = tmp_path / "truth.json"
    _write_json(
        truth,
        {
            "pages": {
                "sample": {
                    "regions": [
                        {
                            "truth_id": "sample-T1",
                            "source_bbox_normalized": [0.1, 0.1, 0.5, 0.5],
                        }
                    ]
                }
            }
        },
    )
    output = tmp_path / "output"

    exit_code = replay.main(
        [
            "--archive",
            str(archive),
            "--truth-regions",
            str(truth),
            "--cross-cv-config",
            str(CONFIG_PATH),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    page_output = output / "run-a" / "pages" / "sample"
    union = json.loads(
        (page_output / "replay-anchor-union.json").read_text("utf-8")
    )
    audit = json.loads(
        (page_output / "replay-anchor-rescue-audit.json").read_text("utf-8")
    )
    comparison = json.loads(
        (page_output / "replay-anchor-truth-comparison.json").read_text("utf-8")
    )
    summary = json.loads((output / "summary.json").read_text("utf-8"))
    assert union[0]["source"] == "independent_scan_rescue"
    assert union[0]["cross_id"] == 0
    assert audit["rescued_count"] == 1
    assert comparison["matched_truth_ids"] == ["sample-T1"]
    assert summary[0]["added_false_anchor_count"] == 0
    assert summary[0]["recovered_truth_ids"] == ["sample-T1"]
    assert (output / "replay-report.md").is_file()


def test_safe_extract_tar_rejects_parent_traversal(tmp_path):
    replay = _load_replay_module()
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"unsafe"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError, match="unsafe tar member"):
            replay._safe_extract_tar(tar, tmp_path / "extract")
