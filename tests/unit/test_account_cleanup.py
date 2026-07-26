from datetime import datetime
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql


class ScalarRows:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class ExecuteRows:
    def __init__(self, values=()):
        self.values = values

    def all(self):
        return list(self.values)


class CleanupDB:
    def __init__(self, *, accounts=(), jobs=(), returned_rows=None):
        self.scalar_results = [accounts, jobs]
        self.returned_rows = returned_rows or {}
        self.scalar_statements = []
        self.execute_statements = []
        self.added = []
        self.deleted = []
        self.commit_calls = 0

    def scalars(self, statement):
        self.scalar_statements.append(statement)
        values = self.scalar_results.pop(0) if self.scalar_results else ()
        return ScalarRows(values)

    def execute(self, statement):
        self.execute_statements.append(statement)
        table_name = statement.table.name
        return ExecuteRows(self.returned_rows.get(table_name, ()))

    def add_all(self, values):
        self.added.extend(values)

    def delete(self, value):
        self.deleted.append(value)

    def commit(self):
        self.commit_calls += 1


def _settings():
    return SimpleNamespace(
        ACCOUNT_CLEANUP_BATCH_SIZE=50,
        ACCOUNT_CLEANUP_INTERVAL_SECONDS=86_400,
    )


def test_cleanup_query_claims_only_due_pending_accounts_in_locked_batch(
    monkeypatch,
):
    from app.services import account_cleanup

    db = CleanupDB()
    monkeypatch.setattr(account_cleanup, "settings", _settings())

    summary = account_cleanup.cleanup_expired_accounts(
        db,
        now=datetime(2026, 7, 26, 8, 30),
    )

    assert summary.accounts_deleted == 0
    statement = db.scalar_statements[0]
    sql = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    assert "accounts.status = 'pending_deletion'" in sql
    assert "accounts.deletion_due_at IS NOT NULL" in sql
    assert "accounts.deletion_due_at <= '2026-07-26 08:30:00'" in sql
    assert "ORDER BY accounts.deletion_due_at, accounts.id" in sql
    assert "LIMIT 50" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql


def test_account_data_is_deleted_in_dependency_order_and_files_are_queued():
    from app.services.account_cleanup import _delete_account_data

    db = CleanupDB(
        returned_rows={
            "practice_sheets": [("/pdfs/sheet.pdf",)],
            "wrong_images": [("/uploads/image.jpg",)],
        }
    )
    account = SimpleNamespace(
        id="account-1",
        avatar_object_key="/avatars/avatar.jpg",
    )

    queued = _delete_account_data(db, account)

    assert [statement.table.name for statement in db.execute_statements] == [
        "practice_results",
        "practice_attempts",
        "sheet_items",
        "practice_sheets",
        "wrong_questions",
        "wrong_images",
        "students",
        "wechat_identities",
        "accounts",
    ]
    assert {(item.storage_kind, item.object_path) for item in queued} == {
        ("avatar", "/avatars/avatar.jpg"),
        ("pdf", "/pdfs/sheet.pdf"),
        ("upload", "/uploads/image.jpg"),
    }


def test_failed_file_cleanup_retains_neutral_retry_job(monkeypatch):
    from app.services import account_cleanup

    job = SimpleNamespace(
        object_path="/avatars/avatar.jpg",
        attempt_count=0,
        last_error=None,
        next_attempt_at=None,
    )
    db = CleanupDB()
    db.scalar_results = [[job]]
    monkeypatch.setattr(account_cleanup, "settings", _settings())

    def fail_cleanup(_job):
        raise PermissionError("do not persist this filesystem detail")

    monkeypatch.setattr(account_cleanup, "_delete_cleanup_path", fail_cleanup)

    summary = account_cleanup._retry_file_cleanup_jobs(
        db,
        cleanup_time=datetime(2026, 7, 26, 8, 30),
    )

    assert summary == (0, 1)
    assert db.deleted == []
    assert job.attempt_count == 1
    assert job.last_error == "PermissionError"
    assert job.next_attempt_at == datetime(2026, 7, 27, 8, 30)


def test_cleanup_is_idempotent_when_no_accounts_or_jobs(monkeypatch):
    from app.services import account_cleanup

    db = CleanupDB()
    monkeypatch.setattr(account_cleanup, "settings", _settings())

    first = account_cleanup.cleanup_expired_accounts(db)
    db.scalar_results = [(), ()]
    second = account_cleanup.cleanup_expired_accounts(db)

    assert first == second
    assert second.accounts_deleted == 0
    assert second.files_queued == 0
    assert second.files_deleted == 0
    assert second.file_failures == 0
    assert db.deleted == []


def test_cleanup_processes_all_due_account_batches(monkeypatch):
    from app.services import account_cleanup

    first_account = SimpleNamespace(id="account-1")
    second_account = SimpleNamespace(id="account-2")
    db = CleanupDB()
    db.scalar_results = [[first_account], [second_account], []]
    monkeypatch.setattr(account_cleanup, "settings", _settings())
    deleted_accounts = []
    monkeypatch.setattr(
        account_cleanup,
        "_delete_account_data",
        lambda _db, account: deleted_accounts.append(account.id) or [],
    )
    monkeypatch.setattr(
        account_cleanup,
        "_retry_file_cleanup_jobs",
        lambda _db, cleanup_time: (0, 0),
    )

    summary = account_cleanup.cleanup_expired_accounts(db)

    assert deleted_accounts == ["account-1", "account-2"]
    assert summary.accounts_deleted == 2
    assert db.commit_calls == 2


def test_file_cleanup_failure_does_not_starve_later_jobs(monkeypatch):
    from app.services import account_cleanup

    failed_job = SimpleNamespace(
        object_path="/avatars/failed.jpg",
        attempt_count=0,
        last_error=None,
        next_attempt_at=None,
    )
    successful_job = SimpleNamespace(
        object_path="/avatars/success.jpg",
        attempt_count=0,
        last_error=None,
        next_attempt_at=None,
    )
    db = CleanupDB()
    db.scalar_results = [[failed_job], [successful_job], []]
    cleanup_settings = _settings()
    cleanup_settings.ACCOUNT_CLEANUP_BATCH_SIZE = 1
    monkeypatch.setattr(account_cleanup, "settings", cleanup_settings)

    def delete_path(job):
        if job is failed_job:
            raise PermissionError("persistent failure")

    monkeypatch.setattr(account_cleanup, "_delete_cleanup_path", delete_path)

    summary = account_cleanup._retry_file_cleanup_jobs(
        db,
        cleanup_time=datetime(2026, 7, 26, 8, 30),
    )

    assert summary == (1, 1)
    assert db.deleted == [successful_job]
    assert failed_job.next_attempt_at == datetime(2026, 7, 27, 8, 30)
    assert db.commit_calls == 2
