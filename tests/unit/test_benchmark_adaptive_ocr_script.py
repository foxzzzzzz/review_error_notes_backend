import os
import subprocess
import sys
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "benchmark_adaptive_ocr.py"


def test_benchmark_script_resolves_backend_app_when_run_by_file_path(tmp_path):
    if sys.version_info < (3, 10):
        pytest.skip("backend scripts require Python 3.10 or newer")

    resource_stub = tmp_path / "resource.py"
    resource_stub.write_text(
        "RUSAGE_SELF = 0\n"
        "def getrusage(_target):\n"
        "    return type('Usage', (), {'ru_maxrss': 0})()\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr
    assert "--repeats" in result.stdout


def test_api_test_container_mounts_repository_artifacts():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    api = compose.split("  api:", 1)[1].split("  worker:", 1)[0]

    assert "./README.md:/app/README.md:ro" in api
    assert "./Dockerfile:/app/Dockerfile:ro" in api
    assert "./scripts:/app/scripts:ro" in api
