import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def disable_real_task_dispatch(monkeypatch):
    from app.api import upload as upload_api

    monkeypatch.setattr(upload_api.process_image, "delay", lambda *_args: None)


class MemoryUpload:
    filename = "question.jpg"

    async def read(self):
        return b"test-image"


class UploadDB:
    def __init__(self, fail_first_commit=False):
        self.fail_first_commit = fail_first_commit
        self.commit_calls = 0
        self.rollback_calls = 0
        self.deleted = []

    def add(self, _value):
        pass

    async def commit(self):
        self.commit_calls += 1
        if self.fail_first_commit and self.commit_calls == 1:
            raise RuntimeError("database unavailable")

    async def rollback(self):
        self.rollback_calls += 1

    async def refresh(self, value):
        value.id = "image-id"

    async def delete(self, value):
        self.deleted.append(value)


def run_upload(db, student, **kwargs):
    from app.api.upload import upload_image

    return asyncio.run(
        upload_image(
            file=MemoryUpload(),
            subject=None,
            grade=kwargs.get("grade"),
            semester=kwargs.get("semester"),
            student=student,
            db=db,
        )
    )


def test_upload_rejects_unset_grade_and_semester_before_writing(tmp_path, monkeypatch):
    from app.api import upload as upload_api

    monkeypatch.setattr(upload_api.settings, "UPLOAD_DIR", str(tmp_path))

    with pytest.raises(HTTPException) as exc_info:
        run_upload(
            UploadDB(),
            SimpleNamespace(
                id="student-id",
                grade=None,
                semester=None,
            ),
        )

    assert exc_info.value.status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_upload_removes_file_when_database_persistence_fails(tmp_path, monkeypatch):
    from app.api import upload as upload_api

    monkeypatch.setattr(upload_api.settings, "UPLOAD_DIR", str(tmp_path))
    db = UploadDB(fail_first_commit=True)

    with pytest.raises(RuntimeError, match="database unavailable"):
        run_upload(
            db,
            SimpleNamespace(id="student-id", grade=1, semester=1),
        )

    assert db.rollback_calls == 1
    assert list(tmp_path.iterdir()) == []


def test_upload_removes_record_and_file_when_task_dispatch_fails(
    tmp_path,
    monkeypatch,
):
    from app.api import upload as upload_api

    monkeypatch.setattr(upload_api.settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(
        upload_api.process_image,
        "delay",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("redis unavailable")),
    )
    db = UploadDB()

    with pytest.raises(RuntimeError, match="redis unavailable"):
        run_upload(
            db,
            SimpleNamespace(id="student-id", grade=1, semester=1),
        )

    assert db.deleted
    assert db.commit_calls == 2
    assert list(tmp_path.iterdir()) == []
