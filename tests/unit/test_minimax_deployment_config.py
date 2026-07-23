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
        "MINIMAX_LOCALIZATION_CONFIDENCE_THRESHOLD",
        "MINIMAX_LOCALIZATION_MIN_IOU",
        "MINIMAX_IMAGE_MAX_EDGE",
        "MINIMAX_IMAGE_JPEG_QUALITY",
        "TAG_ALIAS_CONFIG_PATH",
    ]
    worker = compose.split("  worker:", 1)[1]
    for name in expected:
        assert f"{name}: ${{{name}" in worker

    assert "sk-" not in compose
    assert "secret-token" not in compose
    assert "./config:/app/config:ro" in worker
    assert "TAG_ALIAS_CONFIG_PATH: ${TAG_ALIAS_CONFIG_PATH:-/app/config/tag-aliases.json}" in worker


def test_api_receives_deepseek_settings_for_synchronous_derivative_generation():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    api = compose.split("  api:", 1)[1].split("  worker:", 1)[0]

    assert "LLM_API_KEY: ${LLM_API_KEY}" in api
    assert "LLM_API_BASE: ${LLM_API_BASE:-https://api.deepseek.com/v1}" in api
    assert "LLM_MODEL: ${LLM_MODEL:-deepseek-v4-pro}" in api


def test_paddleocr_is_absent_from_runtime_and_production_task():
    heavy_requirements = (BACKEND_ROOT / "requirements-heavy.txt").read_text(encoding="utf-8").lower()
    task = (BACKEND_ROOT / "app" / "tasks" / "process_image.py").read_text(encoding="utf-8").lower()

    assert "paddle" not in heavy_requirements
    assert "paddle" not in task
    assert "recognize_text" not in task
    assert "segment_questions" not in task


def test_docker_uses_configurable_tsinghua_debian_mirror_before_apt_update():
    dockerfile = (BACKEND_ROOT / "Dockerfile").read_text(encoding="utf-8")

    arg = "ARG DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn"
    replacement = 's|http://deb.debian.org|${DEBIAN_MIRROR}|g'
    assert arg in dockerfile
    assert replacement in dockerfile
    assert dockerfile.index(arg) < dockerfile.index(replacement) < dockerfile.index("apt-get update")
