from pathlib import Path


BACKEND_ROOT = Path(__file__).parents[2]


def test_worker_receives_every_minimax_setting_without_a_secret_value():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    expected = [
        "MINIMAX_API_KEY",
        "MINIMAX_API_HOST",
        "MINIMAX_VISION_TIMEOUT_SECONDS",
        "MINIMAX_VISION_MAX_RETRIES",
        "MINIMAX_VISION_RETRY_DELAY_SECONDS",
        "MINIMAX_CONFIDENCE_THRESHOLD",
        "MINIMAX_IMAGE_MAX_EDGE",
        "MINIMAX_IMAGE_JPEG_QUALITY",
    ]
    worker = compose.split("  worker:", 1)[1]
    for name in expected:
        assert f"{name}: ${{{name}" in worker

    assert "sk-" not in compose
    assert "secret-token" not in compose


def test_paddleocr_is_absent_from_runtime_and_production_task():
    heavy_requirements = (BACKEND_ROOT / "requirements-heavy.txt").read_text(encoding="utf-8").lower()
    task = (BACKEND_ROOT / "app" / "tasks" / "process_image.py").read_text(encoding="utf-8").lower()

    assert "paddle" not in heavy_requirements
    assert "paddle" not in task
    assert "recognize_text" not in task
    assert "segment_questions" not in task
