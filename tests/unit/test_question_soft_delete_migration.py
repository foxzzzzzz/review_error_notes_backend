import ast
from pathlib import Path


BACKEND_ROOT = Path(__file__).parents[2]
MIGRATION_PATH = BACKEND_ROOT / "alembic" / "versions" / "0003_add_question_soft_delete.py"


def _calls(source: str, method: str):
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
    ]


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.unparse(function)


def _constant_list(node: ast.AST) -> list[str]:
    assert isinstance(node, ast.List)
    return [item.value for item in node.elts if isinstance(item, ast.Constant)]


def _foreign_keys(source: str) -> dict[str, dict[str, object]]:
    foreign_keys = {}
    for call in _calls(source, "create_foreign_key"):
        assert len(call.args) >= 5
        assert all(isinstance(call.args[index], ast.Constant) for index in range(3))
        foreign_keys[call.args[0].value] = {
            "source_table": call.args[1].value,
            "target_table": call.args[2].value,
            "local_columns": _constant_list(call.args[3]),
            "target_columns": _constant_list(call.args[4]),
            "ondelete": next(
                (keyword.value.value for keyword in call.keywords if keyword.arg == "ondelete"),
                None,
            ),
        }
    return foreign_keys


def _alter_column_nullable(source: str, expected: bool) -> bool:
    return any(
        len(call.args) >= 2
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "sheet_items"
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "wrong_question_id"
        and any(
            keyword.arg == "nullable"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is expected
            for keyword in call.keywords
        )
        for call in _calls(source, "alter_column")
    )


def test_soft_delete_migration_adds_deleted_at_column_and_index():
    source = MIGRATION_PATH.read_text(encoding="utf-8")

    assert "down_revision" in source and '"0002"' in source
    assert any(
        len(call.args) >= 2
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "wrong_questions"
        and isinstance(call.args[1], ast.Call)
        and isinstance(call.args[1].func, ast.Attribute)
        and call.args[1].func.attr == "Column"
        and call.args[1].args[0].value == "deleted_at"
        for call in _calls(source, "add_column")
    )
    assert "ix_wrong_questions_deleted_at" in source


def test_soft_delete_migration_preserves_sheet_history_when_questions_are_purged():
    source = MIGRATION_PATH.read_text(encoding="utf-8")
    upgrade = _function_source(source, "upgrade")

    assert _alter_column_nullable(upgrade, expected=True)
    assert _foreign_keys(upgrade) == {
        "fk_sheet_items_wrong_question_id": {
            "source_table": "sheet_items",
            "target_table": "wrong_questions",
            "local_columns": ["wrong_question_id"],
            "target_columns": ["id"],
            "ondelete": "SET NULL",
        },
        "fk_sheet_items_derived_from": {
            "source_table": "sheet_items",
            "target_table": "wrong_questions",
            "local_columns": ["derived_from"],
            "target_columns": ["id"],
            "ondelete": "SET NULL",
        },
    }


def test_soft_delete_migration_downgrade_restores_existing_constraints():
    source = MIGRATION_PATH.read_text(encoding="utf-8")
    downgrade = _function_source(source, "downgrade")

    assert "ondelete" not in downgrade
    assert _alter_column_nullable(downgrade, expected=False)
    assert "DELETE FROM sheet_items" not in downgrade
    assert any(
        call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "ix_wrong_questions_deleted_at"
        for call in _calls(downgrade, "drop_index")
    )


def test_models_match_soft_delete_schema():
    question_model = (BACKEND_ROOT / "app" / "models" / "wrong_question.py").read_text(encoding="utf-8")
    sheet_item_model = (BACKEND_ROOT / "app" / "models" / "sheet_item.py").read_text(encoding="utf-8")

    assert "deleted_at = Column(DateTime, nullable=True, index=True)" in question_model
    assert sheet_item_model.count('ForeignKey("wrong_questions.id", ondelete="SET NULL")') == 2
    assert sheet_item_model.count("nullable=True") >= 2
