import importlib.util
from pathlib import Path


def _load_migration():
    path = Path(__file__).parents[2] / "alembic" / "versions" / "0003_image_processing_status.py"
    spec = importlib.util.spec_from_file_location("image_processing_status_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingOp:
    def __init__(self):
        self.commands = []
        self.dropped_columns = []

    def execute(self, command):
        self.commands.append(command)

    def drop_column(self, table_name, column_name):
        self.dropped_columns.append((table_name, column_name))


def test_downgrade_replaces_failed_status_before_restoring_old_enum(monkeypatch):
    migration = _load_migration()
    op = RecordingOp()
    monkeypatch.setattr(migration, "op", op)

    migration.downgrade()

    commands = "\n".join(op.commands)
    assert "UPDATE wrong_images SET status = 'pending' WHERE status = 'failed'" in commands
    assert "CREATE TYPE image_status_enum_without_failed" in commands
    assert "ALTER COLUMN status TYPE image_status_enum_without_failed" in commands
    assert "DROP TYPE image_status_enum" in commands
    assert "RENAME TO image_status_enum" in commands
    assert op.dropped_columns == [
        ("wrong_images", "error_message"),
        ("wrong_images", "error_code"),
    ]
