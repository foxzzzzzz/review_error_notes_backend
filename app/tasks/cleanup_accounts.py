"""Celery task for expired account cleanup."""

from dataclasses import asdict

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.services.account_cleanup import cleanup_expired_accounts
from app.tasks.celery_app import celery_app


@celery_app.task(
    name="app.tasks.cleanup_accounts.cleanup_expired_accounts_task"
)
def cleanup_expired_accounts_task():
    sync_url = settings.DATABASE_URL.replace(
        "postgresql+asyncpg://",
        "postgresql+psycopg2://",
    )
    engine = create_engine(sync_url)
    try:
        with Session(engine) as db:
            return asdict(cleanup_expired_accounts(db))
    finally:
        engine.dispose()
