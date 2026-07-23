class FakeSession:
    def __init__(self):
        self.deleted_tables = []
        self.commits = 0

    def execute(self, statement):
        self.deleted_tables.append(statement.table.name)

    def commit(self):
        self.commits += 1


def test_reset_business_records_preserves_students():
    from app.maintenance.reset_debug_data import reset_business_records

    session = FakeSession()

    reset_business_records(session)

    assert session.deleted_tables == [
        "sheet_items",
        "practice_sheets",
        "wrong_questions",
        "wrong_images",
    ]
    assert "students" not in session.deleted_tables
    assert session.commits == 1


def test_clear_storage_files_only_removes_files_in_target_directory(tmp_path):
    from app.maintenance.reset_debug_data import clear_storage_files

    upload = tmp_path / "uploads"
    upload.mkdir()
    (upload / "question.jpg").write_bytes(b"image")
    nested = upload / "keep-directory"
    nested.mkdir()
    (nested / "keep.txt").write_text("keep", encoding="utf-8")

    removed = clear_storage_files(str(upload))

    assert removed == 1
    assert list(upload.iterdir()) == [nested]
    assert (nested / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_clear_storage_files_rejects_an_unexpected_directory(tmp_path):
    import pytest

    from app.maintenance.reset_debug_data import clear_storage_files

    unsafe = tmp_path / "application"
    unsafe.mkdir()
    (unsafe / "keep.env").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="uploads or pdfs"):
        clear_storage_files(str(unsafe))

    assert (unsafe / "keep.env").read_text(encoding="utf-8") == "keep"


def test_confirmation_must_match_configured_phrase():
    from app.maintenance.reset_debug_data import confirmation_matches

    assert confirmation_matches("CLEAR_DEBUG_BUSINESS_DATA", "CLEAR_DEBUG_BUSINESS_DATA")
    assert not confirmation_matches("yes", "CLEAR_DEBUG_BUSINESS_DATA")
    assert not confirmation_matches("", "CLEAR_DEBUG_BUSINESS_DATA")
