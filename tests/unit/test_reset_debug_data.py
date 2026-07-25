import pytest


class FakeSession:
    def __init__(self):
        self.deleted_tables = []
        self.commits = 0

    def execute(self, statement):
        self.deleted_tables.append(statement.table.name)

    def commit(self):
        self.commits += 1


def test_full_reset_removes_business_and_account_records_in_fk_order():
    from app.maintenance.reset_debug_data import reset_all_test_records

    session = FakeSession()
    reset_all_test_records(session)

    assert session.deleted_tables == [
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
    assert session.commits == 1


@pytest.mark.parametrize("app_env", ["production", "staging", ""])
def test_full_reset_requires_test_environment(app_env):
    from app.maintenance.reset_debug_data import assert_reset_allowed

    with pytest.raises(RuntimeError, match="test environment"):
        assert_reset_allowed(
            app_env,
            "RESET_ALL_TEST_DATA",
            "RESET_ALL_TEST_DATA",
        )


def test_full_reset_requires_exact_confirmation():
    from app.maintenance.reset_debug_data import assert_reset_allowed

    with pytest.raises(ValueError, match="confirmation phrase"):
        assert_reset_allowed("development", "yes", "RESET_ALL_TEST_DATA")

    assert_reset_allowed(
        "test",
        "RESET_ALL_TEST_DATA",
        "RESET_ALL_TEST_DATA",
    )


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


@pytest.mark.parametrize("safe_name", ["uploads", "pdfs", "avatars"])
def test_clear_storage_files_accepts_only_named_storage_roots(tmp_path, safe_name):
    from app.maintenance.reset_debug_data import clear_storage_files

    storage = tmp_path / safe_name
    storage.mkdir()

    assert clear_storage_files(str(storage)) == 0


def test_clear_storage_files_rejects_an_unexpected_directory(tmp_path):
    from app.maintenance.reset_debug_data import clear_storage_files

    unsafe = tmp_path / "application"
    unsafe.mkdir()
    (unsafe / "keep.env").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="uploads, pdfs, or avatars"):
        clear_storage_files(str(unsafe))

    assert (unsafe / "keep.env").read_text(encoding="utf-8") == "keep"
