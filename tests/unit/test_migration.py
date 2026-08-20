import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
VERSIONS = ROOT / "alembic" / "versions"
EXPECTED_TABLES = {
    "accounts",
    "wechat_identities",
    "students",
    "wrong_images",
    "wrong_questions",
    "practice_sheets",
    "sheet_items",
    "practice_attempts",
    "practice_results",
}


def test_initial_revision_creates_and_drops_every_model_table():
    initial_revision = VERSIONS / "0001_initial_schema.py"
    assert initial_revision.exists()

    tree = ast.parse(initial_revision.read_text(encoding="utf-8"))
    created = set()
    dropped = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.func.attr == "create_table":
            created.add(node.args[0].value)
        elif node.func.attr == "drop_table":
            dropped.add(node.args[0].value)

    assert created == EXPECTED_TABLES
    assert dropped == EXPECTED_TABLES


def test_alembic_environment_imports_every_model_module():
    source = (ROOT / "alembic" / "env.py").read_text(encoding="utf-8")

    for module in (
        "account",
        "wechat_identity",
        "student",
        "wrong_image",
        "wrong_question",
        "practice_sheet",
        "sheet_item",
        "practice_attempt",
        "practice_result",
    ):
        assert f"app.models.{module}" in source


def test_initial_schema_contains_account_identity_and_mastery_fields():
    source = (VERSIONS / "0001_initial_schema.py").read_text(encoding="utf-8")

    assert 'op.create_table(\n        "accounts"' in source
    assert 'op.create_table(\n        "wechat_identities"' in source
    assert 'sa.Column("account_id"' in source
    assert '"review_status",' in source
    assert '"mastery_status",' in source
    assert '"uq_wechat_identity_appid_openid"' in source
    assert '"uq_attempt_student_key"' in source
    assert '"uq_attempt_sheet_number"' in source
    assert '"uq_result_attempt_item"' in source
    assert 'sa.Column("question_snapshot", postgresql.JSONB' in source
    assert '"ix_practice_results_wrong_question_id"' in source


def test_runtime_processes_leave_schema_management_to_alembic():
    for runtime_file in (
        ROOT / "app" / "main.py",
        ROOT / "app" / "tasks" / "process_image.py",
    ):
        source = runtime_file.read_text(encoding="utf-8")
        assert "create_all" not in source, (
            f"{runtime_file.relative_to(ROOT)} must not create database tables at runtime"
        )


def test_phase_four_recovery_support_revision_adds_audit_and_cleanup_tables():
    revision = VERSIONS / "0002_phase4_recovery_support.py"

    assert revision.exists()
    source = revision.read_text(encoding="utf-8")
    assert 'op.create_table(\n        "account_recovery_conflicts"' in source
    assert 'op.create_table(\n        "file_cleanup_jobs"' in source
    assert 'down_revision: Union[str, None] = "0001"' in source


def test_alembic_environment_imports_phase_four_recovery_models():
    source = (ROOT / "alembic" / "env.py").read_text(encoding="utf-8")

    assert "app.models.account_recovery" in source


def test_sheet_duration_revision_follows_async_generation_revision():
    revision = VERSIONS / "0006_sheet_duration.py"

    assert revision.exists()
    source = revision.read_text(encoding="utf-8")
    assert 'down_revision: Union[str, None] = "0005"' in source
    assert '"generation_started_at"' in source
    assert '"generation_duration_seconds"' in source


def test_image_recognition_correction_revision_follows_sheet_duration():
    revision = VERSIONS / "0007_image_recognition_correction.py"

    assert revision.exists()
    source = revision.read_text(encoding="utf-8")
    assert 'down_revision: Union[str, None] = "0006"' in source
    assert '"recognition_correction"' in source
