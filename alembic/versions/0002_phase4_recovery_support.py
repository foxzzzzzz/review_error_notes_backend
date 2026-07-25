"""Add account recovery audit and file cleanup jobs.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _timestamps():
    return (
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def upgrade() -> None:
    op.create_table(
        "account_recovery_conflicts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("support_reference", sa.String(length=15), nullable=False),
        sa.Column(
            "current_account_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "target_account_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="open",
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('open', 'resolved')",
            name="ck_account_recovery_conflicts_status",
        ),
        sa.ForeignKeyConstraint(
            ["current_account_id"],
            ["accounts.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["target_account_id"],
            ["accounts.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_account_recovery_conflicts_support_reference",
        "account_recovery_conflicts",
        ["support_reference"],
        unique=True,
    )
    op.create_index(
        "ix_account_recovery_conflicts_current_account_id",
        "account_recovery_conflicts",
        ["current_account_id"],
        unique=False,
    )
    op.create_index(
        "ix_account_recovery_conflicts_target_account_id",
        "account_recovery_conflicts",
        ["target_account_id"],
        unique=False,
    )
    op.create_index(
        "ix_account_recovery_conflicts_status",
        "account_recovery_conflicts",
        ["status"],
        unique=False,
    )

    op.create_table(
        "file_cleanup_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("storage_kind", sa.String(length=32), nullable=False),
        sa.Column("object_path", sa.String(length=500), nullable=False),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column("last_error", sa.String(length=255), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "storage_kind IN ('avatar', 'upload', 'pdf')",
            name="ck_file_cleanup_jobs_storage_kind",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_file_cleanup_jobs_next_attempt_at",
        "file_cleanup_jobs",
        ["next_attempt_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_file_cleanup_jobs_next_attempt_at",
        table_name="file_cleanup_jobs",
    )
    op.drop_table("file_cleanup_jobs")
    op.drop_index(
        "ix_account_recovery_conflicts_status",
        table_name="account_recovery_conflicts",
    )
    op.drop_index(
        "ix_account_recovery_conflicts_target_account_id",
        table_name="account_recovery_conflicts",
    )
    op.drop_index(
        "ix_account_recovery_conflicts_current_account_id",
        table_name="account_recovery_conflicts",
    )
    op.drop_index(
        "ix_account_recovery_conflicts_support_reference",
        table_name="account_recovery_conflicts",
    )
    op.drop_table("account_recovery_conflicts")
