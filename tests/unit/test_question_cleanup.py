from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from app.models.sheet_item import SheetItem
from app.services import question_cleanup as cleanup_module


class Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class CapturingSession:
    def __init__(self, result_batches):
        self.result_batches = iter(result_batches)
        self.queries = []
        self.deleted = []
        self.commit_calls = 0
        self.rollback_calls = 0

    def scalars(self, query):
        self.queries.append(query)
        return Result(next(self.result_batches))

    def delete(self, record):
        self.deleted.append(record)

    def commit(self):
        self.commit_calls += 1

    def rollback(self):
        self.rollback_calls += 1


def _compiled(query):
    return query.compile(dialect=postgresql.dialect())


def _settings(upload_dir):
    return SimpleNamespace(
        QUESTION_SOFT_DELETE_RETENTION_DAYS=30,
        QUESTION_CLEANUP_BATCH_SIZE=100,
        UPLOAD_DIR=str(upload_dir),
    )


def test_cleanup_selects_only_expired_soft_deleted_questions_in_locked_batches(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cleanup_module, "settings", _settings(tmp_path))
    db = CapturingSession([[], []])

    cleanup_module.cleanup_expired_questions(
        db,
        now=datetime(2026, 7, 23, 12, 0, 0),
    )

    compiled = _compiled(db.queries[0])
    sql = str(compiled)
    assert "wrong_questions.deleted_at IS NOT NULL" in sql
    assert "wrong_questions.deleted_at <" in sql
    assert "LIMIT" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "ORDER BY wrong_questions.deleted_at, wrong_questions.id" in sql
    assert datetime(2026, 6, 23, 12, 0, 0) in compiled.params.values()
    assert 100 in compiled.params.values()


def test_physical_question_deletion_relies_on_set_null_snapshot_foreign_keys(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cleanup_module, "settings", _settings(tmp_path))
    question = SimpleNamespace(id="question-id")
    db = CapturingSession([[question], [], []])

    result = cleanup_module.cleanup_expired_questions(
        db,
        now=datetime(2026, 7, 23, 12, 0, 0),
    )

    assert db.deleted == [question]
    assert result["questions_deleted"] == 1
    foreign_key = next(iter(SheetItem.wrong_question_id.property.columns[0].foreign_keys))
    assert foreign_key.ondelete == "SET NULL"


def test_orphan_query_requires_no_question_reference_and_an_expired_image(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cleanup_module, "settings", _settings(tmp_path))
    db = CapturingSession([[], []])

    cleanup_module.cleanup_expired_questions(
        db,
        now=datetime(2026, 7, 23, 12, 0, 0),
    )

    compiled = _compiled(db.queries[1])
    sql = str(compiled)
    assert "wrong_images.created_at <" in sql
    assert "NOT (EXISTS" in sql
    assert "wrong_questions.image_id = wrong_images.id" in sql
    assert "ORDER BY wrong_images.created_at, wrong_images.id" in sql
    assert datetime(2026, 6, 23, 12, 0, 0) in compiled.params.values()


def test_missing_orphan_file_is_success_and_deletes_image_record(monkeypatch, tmp_path):
    monkeypatch.setattr(cleanup_module, "settings", _settings(tmp_path))
    image = SimpleNamespace(id="image-id", original_url="/uploads/missing.jpg")
    db = CapturingSession([[], [image], []])

    result = cleanup_module.cleanup_expired_questions(
        db,
        now=datetime(2026, 7, 23, 12, 0, 0),
    )

    assert not (tmp_path / "missing.jpg").exists()
    assert db.deleted == [image]
    assert result["images_deleted"] == 1


def test_filesystem_error_keeps_image_record_for_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(cleanup_module, "settings", _settings(tmp_path))
    image = SimpleNamespace(id="image-id", original_url="/uploads/retry.jpg")
    db = CapturingSession([[], [image], []])

    def fail_unlink(_self, *, missing_ok):
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    result = cleanup_module.cleanup_expired_questions(
        db,
        now=datetime(2026, 7, 23, 12, 0, 0),
    )

    assert image not in db.deleted
    assert db.rollback_calls == 0
    assert db.commit_calls == 1
    assert result["images_deleted"] == 0


def test_filesystem_error_keeps_lock_and_does_not_block_later_images(monkeypatch, tmp_path):
    monkeypatch.setattr(cleanup_module, "settings", _settings(tmp_path))
    failed_image = SimpleNamespace(id="failed", original_url="/uploads/failed.jpg")
    later_image = SimpleNamespace(id="later", original_url="/uploads/later.jpg")
    db = CapturingSession([[], [failed_image, later_image], []])
    unlink_calls = []

    def unlink(self, *, missing_ok):
        unlink_calls.append(self.name)
        if self.name == "failed.jpg":
            raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "unlink", unlink)

    result = cleanup_module.cleanup_expired_questions(
        db,
        now=datetime(2026, 7, 23, 12, 0, 0),
    )

    assert unlink_calls == ["failed.jpg", "later.jpg"]
    assert db.deleted == [later_image]
    assert db.rollback_calls == 0
    assert db.commit_calls == 1
    assert result["images_deleted"] == 1
    retry_query = _compiled(db.queries[2])
    assert "wrong_images.id NOT IN" in str(retry_query)
    assert ["failed"] in retry_query.params.values()


def test_cleanup_processes_all_question_and_image_batches(monkeypatch, tmp_path):
    monkeypatch.setattr(cleanup_module, "settings", _settings(tmp_path))
    questions = [SimpleNamespace(id=f"question-{index}") for index in range(3)]
    images = [
        SimpleNamespace(id=f"image-{index}", original_url=f"/uploads/image-{index}.jpg")
        for index in range(2)
    ]
    db = CapturingSession(
        [
            questions[:2],
            questions[2:],
            [],
            images[:1],
            images[1:],
            [],
        ]
    )

    result = cleanup_module.cleanup_expired_questions(
        db,
        now=datetime(2026, 7, 23, 12, 0, 0),
    )

    assert db.deleted == questions + images
    assert db.commit_calls == 4
    assert result == {"questions_deleted": 3, "images_deleted": 2}


def test_aware_cleanup_time_is_converted_to_naive_utc(monkeypatch, tmp_path):
    monkeypatch.setattr(cleanup_module, "settings", _settings(tmp_path))
    db = CapturingSession([[], []])
    china_time = datetime(
        2026,
        7,
        23,
        20,
        0,
        0,
        tzinfo=timezone(timedelta(hours=8)),
    )

    cleanup_module.cleanup_expired_questions(db, now=china_time)

    cutoff = datetime(2026, 6, 23, 12, 0, 0)
    assert cutoff in _compiled(db.queries[0]).params.values()
