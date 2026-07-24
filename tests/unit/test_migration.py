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
