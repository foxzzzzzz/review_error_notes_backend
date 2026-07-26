from pathlib import Path


PROJECT_DIR = Path(__file__).parents[2]


def test_account_cleanup_configuration_defaults():
    from app.config import Settings

    settings = Settings(_env_file=None)

    assert settings.ACCOUNT_CLEANUP_INTERVAL_SECONDS == 86_400
    assert settings.ACCOUNT_CLEANUP_BATCH_SIZE == 50


def test_account_cleanup_is_registered_with_celery_beat():
    source = (PROJECT_DIR / "app" / "tasks" / "celery_app.py").read_text(
        encoding="utf-8"
    )

    assert '"app.tasks.cleanup_accounts"' in source
    assert '"cleanup-expired-accounts"' in source
    assert (
        '"app.tasks.cleanup_accounts.cleanup_expired_accounts_task"' in source
    )
    assert "settings.ACCOUNT_CLEANUP_INTERVAL_SECONDS" in source


def test_worker_has_cleanup_storage_and_configuration():
    compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")

    assert "ACCOUNT_CLEANUP_INTERVAL_SECONDS" in env_example
    assert "ACCOUNT_CLEANUP_BATCH_SIZE" in env_example
    api = compose.split("  api:", 1)[1].split("  worker:", 1)[0]
    worker = compose.split("  worker:", 1)[1].split("  beat:", 1)[0]
    beat = compose.split("  beat:", 1)[1].split("volumes:", 1)[0]

    assert "ACCOUNT_CLEANUP_INTERVAL_SECONDS" not in api
    assert "ACCOUNT_CLEANUP_BATCH_SIZE" not in api
    assert "ACCOUNT_CLEANUP_INTERVAL_SECONDS" in worker
    assert "ACCOUNT_CLEANUP_BATCH_SIZE" in worker
    assert "ACCOUNT_CLEANUP_INTERVAL_SECONDS" in beat
    assert "ACCOUNT_CLEANUP_BATCH_SIZE" not in beat
    assert "PDF_DIR: /app/pdfs" in worker
    assert "AVATAR_DIR: /app/avatars" in worker
    assert "- pdfs:/app/pdfs" in worker
    assert "- avatars:/app/avatars" in worker
