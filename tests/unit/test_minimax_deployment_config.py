from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings


BACKEND_ROOT = Path(__file__).parents[2]

EXPECTED_ADAPTIVE_EVIDENCE_SETTINGS = {
    "MARK_CORRECTION_GROUP_ENABLED": "true",
    "MARK_PAIR_MAX_DISTANCE_RATIO": "0.04",
    "MARK_ANCHOR_MAX_GAP_RATIO": "0.08",
    "MARK_CROSS_ONLY_MAX_GAP_RATIO": "0.08",
    "MARK_DEDUP_IOU_THRESHOLD": "0.8",
    "MINIMAX_LOCALIZATION_SEMANTIC_RETRY_COUNT": "1",
    "LOCAL_OCR_MARKED_RECHECK_LIMIT": "3",
    "LOCAL_OCR_ENABLED": "true",
    "LOCAL_OCR_FULL_PAGE_MAX_EDGE": "1600",
    "LOCAL_OCR_CROP_RECHECK_LIMIT": "3",
    "LOCAL_RED_SCAN_MAX_EDGE": "1600",
    "LOCAL_RED_COMPONENT_MIN_PIXELS": "12",
    "LOCAL_RED_COMPONENT_MAX_AREA_RATIO": "0.08",
    "LOCAL_RED_COMPONENT_MAX_THINNESS_RATIO": "18",
    "LOCAL_RED_RESCUE_MIN_PIXELS": "80",
    "MINIMAX_MARK_MISMATCH_RETRY_COUNT": "1",
    "CELERY_WORKER_CONCURRENCY": "2",
}


def test_worker_receives_every_minimax_setting_without_a_secret_value():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    expected = [
        "MINIMAX_API_KEY",
        "MINIMAX_API_HOST",
        "MINIMAX_VISION_TIMEOUT_SECONDS",
        "MINIMAX_VISION_MAX_RETRIES",
        "MINIMAX_VISION_RETRY_DELAY_SECONDS",
        "MINIMAX_CONFIDENCE_THRESHOLD",
        "MINIMAX_MARK_CONFIDENCE_THRESHOLD",
        "MINIMAX_LOCALIZATION_CONFIDENCE_THRESHOLD",
        "MINIMAX_LOCALIZATION_MAX_AREA_RATIO",
        "QUESTION_CROP_CONTEXT_PADDING_RATIO",
        "MARK_RED_PIXEL_MIN_RATIO",
        "MARK_RED_PIXEL_EXPANSION_RATIO",
        "MINIMAX_IMAGE_MAX_EDGE",
        "MINIMAX_IMAGE_JPEG_QUALITY",
        "LOCAL_OCR_ENABLED",
        "LOCAL_OCR_ENGINE",
        "LOCAL_OCR_VERSION",
        "LOCAL_OCR_MODEL_VERSION",
        "LOCAL_OCR_MODEL_TYPE",
        "LOCAL_OCR_MODEL_PATH",
        "LOCAL_OCR_LINE_CONFIDENCE_THRESHOLD",
        "LOCAL_OCR_MIN_EFFECTIVE_CHARACTERS",
        "LOCAL_OCR_SUPPORT_SIMILARITY_THRESHOLD",
        "LOCAL_OCR_CONTRADICTION_SIMILARITY_THRESHOLD",
        "TAG_ALIAS_CONFIG_PATH",
    ]
    worker = compose.split("  worker:", 1)[1]
    for name in expected:
        assert f"{name}: ${{{name}" in worker

    assert "sk-" not in compose
    assert "secret-token" not in compose
    assert "./config:/app/config:ro" in worker
    assert "TAG_ALIAS_CONFIG_PATH: ${TAG_ALIAS_CONFIG_PATH:-/app/config/tag-aliases.json}" in worker


def test_adaptive_local_evidence_settings_have_documented_safe_defaults():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (BACKEND_ROOT / ".env.example").read_text(encoding="utf-8")
    readme = (BACKEND_ROOT / "README.md").read_text(encoding="utf-8")
    config = (BACKEND_ROOT / "app" / "config.py").read_text(encoding="utf-8")
    worker = compose.split("  worker:", 1)[1].split("  beat:", 1)[0]

    for name, default in EXPECTED_ADAPTIVE_EVIDENCE_SETTINGS.items():
        assert name in config
        assert f"{name}={default}" in env_example
        assert f"`{name}`" in readme
        if name != "CELERY_WORKER_CONCURRENCY":
            assert f"{name}: ${{{name}:-{default}}}" in worker

    assert "--concurrency=${CELERY_WORKER_CONCURRENCY:-2}" in worker
    assert "LOCAL_OCR_ENABLED: ${LOCAL_OCR_ENABLED:-false}" not in worker


def test_adaptive_local_evidence_settings_are_bounded():
    settings = Settings(_env_file=None)

    assert settings.MARK_CORRECTION_GROUP_ENABLED is True
    assert settings.MARK_PAIR_MAX_DISTANCE_RATIO == 0.04
    assert settings.MARK_ANCHOR_MAX_GAP_RATIO == 0.08
    assert settings.MARK_CROSS_ONLY_MAX_GAP_RATIO == 0.08
    assert settings.MARK_DEDUP_IOU_THRESHOLD == 0.8
    assert settings.MINIMAX_LOCALIZATION_SEMANTIC_RETRY_COUNT == 1
    assert settings.LOCAL_OCR_MARKED_RECHECK_LIMIT == 3
    assert settings.LOCAL_OCR_ENABLED is True
    assert settings.LOCAL_OCR_FULL_PAGE_MAX_EDGE == 1600
    assert settings.LOCAL_OCR_CROP_RECHECK_LIMIT == 3
    assert settings.LOCAL_RED_RESCUE_MIN_PIXELS == 80
    assert settings.CELERY_WORKER_CONCURRENCY == 2

    with pytest.raises(ValidationError):
        Settings(_env_file=None, LOCAL_OCR_FULL_PAGE_MAX_EDGE=639)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LOCAL_OCR_CROP_RECHECK_LIMIT=21)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LOCAL_RED_COMPONENT_MAX_AREA_RATIO=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LOCAL_RED_RESCUE_MIN_PIXELS=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MARK_PAIR_MAX_DISTANCE_RATIO=1.01)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MARK_DEDUP_IOU_THRESHOLD=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MINIMAX_LOCALIZATION_SEMANTIC_RETRY_COUNT=3)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, CELERY_WORKER_CONCURRENCY=17)


def test_worker_passes_correction_group_settings_to_recognition_batch():
    task = (BACKEND_ROOT / "app" / "tasks" / "process_image.py").read_text(
        encoding="utf-8"
    )

    expected_arguments = {
        "correction_group_enabled": "MARK_CORRECTION_GROUP_ENABLED",
        "pair_max_distance_ratio": "MARK_PAIR_MAX_DISTANCE_RATIO",
        "dedup_iou_threshold": "MARK_DEDUP_IOU_THRESHOLD",
        "anchor_max_gap_ratio": "MARK_ANCHOR_MAX_GAP_RATIO",
        "cross_only_max_gap_ratio": "MARK_CROSS_ONLY_MAX_GAP_RATIO",
        "semantic_retry_count": "MINIMAX_LOCALIZATION_SEMANTIC_RETRY_COUNT",
        "marked_ocr_recheck_limit": "LOCAL_OCR_MARKED_RECHECK_LIMIT",
        "local_red_rescue_min_pixels": "LOCAL_RED_RESCUE_MIN_PIXELS",
    }
    compact_task = "".join(task.split())
    for argument, setting_name in expected_arguments.items():
        assert f"{argument}=settings.{setting_name}" in compact_task


def test_marker_focused_crop_padding_is_external_and_validated():
    settings = Settings(_env_file=None)

    assert settings.QUESTION_CROP_CONTEXT_PADDING_RATIO == 0.15
    with pytest.raises(ValidationError):
        Settings(_env_file=None, QUESTION_CROP_CONTEXT_PADDING_RATIO=-0.01)

    env_example = (BACKEND_ROOT / ".env.example").read_text(encoding="utf-8")
    readme = (BACKEND_ROOT / "README.md").read_text(encoding="utf-8")
    task = (BACKEND_ROOT / "app" / "tasks" / "process_image.py").read_text(
        encoding="utf-8"
    )
    assert "QUESTION_CROP_CONTEXT_PADDING_RATIO=0.15" in env_example
    assert "QUESTION_CROP_CONTEXT_PADDING_RATIO" in readme
    assert (
        "crop_context_padding_ratio=settings.QUESTION_CROP_CONTEXT_PADDING_RATIO"
        in task
    )


def test_api_receives_deepseek_settings_for_synchronous_derivative_generation():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    api = compose.split("  api:", 1)[1].split("  worker:", 1)[0]

    assert "LLM_API_KEY: ${LLM_API_KEY}" in api
    assert "LLM_API_BASE: ${LLM_API_BASE:-https://api.deepseek.com/v1}" in api
    assert "LLM_MODEL: ${LLM_MODEL:-deepseek-v4-pro}" in api


def test_rapidocr_and_onnxruntime_are_pinned_without_paddlepaddle():
    heavy_requirements = (BACKEND_ROOT / "requirements-heavy.txt").read_text(encoding="utf-8").lower()
    task = (BACKEND_ROOT / "app" / "tasks" / "process_image.py").read_text(encoding="utf-8").lower()

    assert "rapidocr==3.9.1" in heavy_requirements
    assert "onnxruntime==1.27.0" in heavy_requirements
    assert "paddlepaddle" not in heavy_requirements
    assert "paddle" not in task
    assert "rapidocrverifier" in task


def test_docker_build_warms_the_configured_ocr_models():
    dockerfile = (BACKEND_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "LOCAL_OCR_MODEL_PATH" in dockerfile
    assert "warm_local_ocr_models.py" in dockerfile


def test_docker_uses_configurable_tsinghua_debian_mirror_before_apt_update():
    dockerfile = (BACKEND_ROOT / "Dockerfile").read_text(encoding="utf-8")

    arg = "ARG DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn"
    replacement = 's|http://deb.debian.org|${DEBIAN_MIRROR}|g'
    assert arg in dockerfile
    assert replacement in dockerfile
    assert dockerfile.index(arg) < dockerfile.index(replacement) < dockerfile.index("apt-get update")
