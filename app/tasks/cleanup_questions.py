"""Celery wrapper for expired wrong-question cleanup."""

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.config import settings
from app.services.question_cleanup import cleanup_expired_questions
from app.tasks.celery_app import celery_app


@celery_app.task(name="app.tasks.cleanup_questions.cleanup_expired_questions_task")
def cleanup_expired_questions_task():
    sync_url = make_url(settings.DATABASE_URL)
    if sync_url.get_backend_name() != "postgresql":
        raise ValueError("Question cleanup requires a PostgreSQL database URL")
    sync_url = sync_url.set(drivername="postgresql+psycopg2")
    engine = create_engine(sync_url)
    try:
        with Session(engine) as db:
            return cleanup_expired_questions(db)
    finally:
        engine.dispose()
