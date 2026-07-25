import asyncio
from types import SimpleNamespace


class CleanupDB:
    def __init__(self, job):
        self.job = job
        self.deleted = []
        self.commit_calls = 0
        self.rollback_calls = 0

    async def scalar(self, _statement):
        return self.job

    async def delete(self, value):
        self.deleted.append(value)

    async def commit(self):
        self.commit_calls += 1

    async def rollback(self):
        self.rollback_calls += 1


def test_successful_file_cleanup_deletes_job(monkeypatch):
    from app.services import file_cleanup

    job = SimpleNamespace(
        id="job-1",
        storage_kind="avatar",
        object_path="/avatars/avatar.jpg",
        attempt_count=0,
        last_error=None,
    )
    db = CleanupDB(job)
    deleted_paths = []
    monkeypatch.setattr(
        file_cleanup,
        "_delete_cleanup_path",
        lambda value: deleted_paths.append(value.object_path),
    )

    result = asyncio.run(file_cleanup.attempt_file_cleanup(db, job.id))

    assert result is True
    assert deleted_paths == ["/avatars/avatar.jpg"]
    assert db.deleted == [job]
    assert db.commit_calls == 1


def test_failed_file_cleanup_retains_retry_job(monkeypatch):
    from app.services import file_cleanup

    job = SimpleNamespace(
        id="job-1",
        storage_kind="avatar",
        object_path="/avatars/avatar.jpg",
        attempt_count=0,
        last_error=None,
    )
    db = CleanupDB(job)

    def fail_cleanup(_job):
        raise PermissionError("sensitive filesystem details")

    monkeypatch.setattr(file_cleanup, "_delete_cleanup_path", fail_cleanup)

    result = asyncio.run(file_cleanup.attempt_file_cleanup(db, job.id))

    assert result is False
    assert db.deleted == []
    assert job.attempt_count == 1
    assert job.last_error == "PermissionError"
    assert db.commit_calls == 1
