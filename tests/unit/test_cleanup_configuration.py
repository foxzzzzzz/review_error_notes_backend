from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import sys

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import URL

from app.config import Settings


BACKEND_ROOT = Path(__file__).parents[2]


def test_cleanup_settings_have_documented_defaults(monkeypatch):
    names = (
        "QUESTION_SOFT_DELETE_RETENTION_DAYS",
        "QUESTION_CLEANUP_INTERVAL_SECONDS",
        "QUESTION_CLEANUP_BATCH_SIZE",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.QUESTION_SOFT_DELETE_RETENTION_DAYS == 30
    assert settings.QUESTION_CLEANUP_INTERVAL_SECONDS == 86_400
    assert settings.QUESTION_CLEANUP_BATCH_SIZE == 100


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("QUESTION_SOFT_DELETE_RETENTION_DAYS", -1),
        ("QUESTION_CLEANUP_INTERVAL_SECONDS", 0),
        ("QUESTION_CLEANUP_BATCH_SIZE", 0),
    ],
)
def test_cleanup_settings_reject_invalid_values(name, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{name: value})


def _load_cleanup_task(monkeypatch):
    celery_module = ModuleType("app.tasks.celery_app")
    celery_module.celery_app = SimpleNamespace(
        task=lambda **_kwargs: lambda function: function
    )
    monkeypatch.setitem(sys.modules, "app.tasks.celery_app", celery_module)
    sys.modules.pop("app.tasks.cleanup_questions", None)
    return importlib.import_module("app.tasks.cleanup_questions")


def test_cleanup_task_passes_a_psycopg2_url_without_rewriting_url_components(
    monkeypatch,
):
    task_module = _load_cleanup_task(monkeypatch)
    task_module.settings = SimpleNamespace(
        DATABASE_URL=(
            "postgresql+asyncpg://user:p+asyncpg@localhost/db"
            "?application_name=cleanup%2Basyncpg"
        )
    )
    captured = {}

    class Engine:
        def dispose(self):
            pass

    class SessionContext:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            pass

    def create_engine(url):
        captured["url"] = url
        return Engine()

    monkeypatch.setattr(task_module, "create_engine", create_engine)
    monkeypatch.setattr(task_module, "Session", lambda _engine: SessionContext())
    monkeypatch.setattr(
        task_module,
        "cleanup_expired_questions",
        lambda _db: {"questions_deleted": 0, "images_deleted": 0},
    )

    task_module.cleanup_expired_questions_task()

    url = captured["url"]
    assert isinstance(url, URL)
    assert url.drivername == "postgresql+psycopg2"
    assert url.password == "p+asyncpg"
    assert url.query["application_name"] == "cleanup+asyncpg"


def test_cleanup_task_rejects_non_postgresql_database_urls(monkeypatch):
    task_module = _load_cleanup_task(monkeypatch)
    task_module.settings = SimpleNamespace(DATABASE_URL="sqlite:///cleanup.db")

    with pytest.raises(ValueError, match="PostgreSQL"):
        task_module.cleanup_expired_questions_task()


def test_celery_registers_cleanup_task_and_daily_beat_schedule():
    source = (BACKEND_ROOT / "app" / "tasks" / "celery_app.py").read_text(encoding="utf-8")

    assert '"app.tasks.cleanup_questions"' in source
    assert '"cleanup-expired-questions"' in source
    assert '"app.tasks.cleanup_questions.cleanup_expired_questions_task"' in source
    assert "settings.QUESTION_CLEANUP_INTERVAL_SECONDS" in source


def test_worker_and_beat_receive_cleanup_settings_and_worker_mounts_uploads():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    worker = compose.split("  worker:", 1)[1].split("  beat:", 1)[0]
    beat = compose.split("  beat:", 1)[1].split("\nvolumes:", 1)[0]

    assert "command: celery -A app.tasks.celery_app beat --loglevel=info" in beat
    assert "uploads:/app/uploads" in worker
    assert "UPLOAD_DIR: /app/uploads" in worker
    for name in (
        "QUESTION_SOFT_DELETE_RETENTION_DAYS",
        "QUESTION_CLEANUP_INTERVAL_SECONDS",
        "QUESTION_CLEANUP_BATCH_SIZE",
    ):
        assert f"{name}: ${{{name}" in worker
        assert f"{name}: ${{{name}" in beat


def test_env_example_explains_every_cleanup_setting():
    env_example = (BACKEND_ROOT / ".env.example").read_text(encoding="utf-8")

    for name in (
        "QUESTION_SOFT_DELETE_RETENTION_DAYS",
        "QUESTION_CLEANUP_INTERVAL_SECONDS",
        "QUESTION_CLEANUP_BATCH_SIZE",
    ):
        setting_line = next(
            index
            for index, line in enumerate(env_example.splitlines())
            if line.startswith(f"{name}=")
        )
        assert env_example.splitlines()[setting_line - 1].startswith("# ")
