"""Add failed image recognition state and safe failure details.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE image_status_enum ADD VALUE IF NOT EXISTS 'failed'")
    op.add_column("wrong_images", sa.Column("error_code", sa.String(length=64), nullable=True))
    op.add_column(
        "wrong_images",
        sa.Column("error_message", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.execute("UPDATE wrong_images SET status = 'pending' WHERE status = 'failed'")
    op.execute("ALTER TABLE wrong_images ALTER COLUMN status DROP DEFAULT")
    op.execute(
        "CREATE TYPE image_status_enum_without_failed "
        "AS ENUM ('pending', 'segmented', 'needs_review', 'confirmed')"
    )
    op.execute(
        "ALTER TABLE wrong_images ALTER COLUMN status "
        "TYPE image_status_enum_without_failed "
        "USING status::text::image_status_enum_without_failed"
    )
    op.execute("DROP TYPE image_status_enum")
    op.execute("ALTER TYPE image_status_enum_without_failed RENAME TO image_status_enum")
    op.execute(
        "ALTER TABLE wrong_images ALTER COLUMN status "
        "SET DEFAULT 'pending'::image_status_enum"
    )
    op.drop_column("wrong_images", "error_message")
    op.drop_column("wrong_images", "error_code")
