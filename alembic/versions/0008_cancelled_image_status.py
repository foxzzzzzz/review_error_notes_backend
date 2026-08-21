"""Add the terminal cancelled status for user-dismissed image tasks.

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-21
"""

from typing import Union

from alembic import op


revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE image_status_enum ADD VALUE IF NOT EXISTS 'cancelled'")


def downgrade() -> None:
    op.execute("UPDATE wrong_images SET status = 'confirmed' WHERE status = 'cancelled'")
    op.execute("ALTER TABLE wrong_images ALTER COLUMN status DROP DEFAULT")
    op.execute(
        "CREATE TYPE image_status_enum_without_cancelled "
        "AS ENUM ('pending', 'segmented', 'needs_review', 'confirmed', 'failed')"
    )
    op.execute(
        "ALTER TABLE wrong_images ALTER COLUMN status "
        "TYPE image_status_enum_without_cancelled "
        "USING status::text::image_status_enum_without_cancelled"
    )
    op.execute("DROP TYPE image_status_enum")
    op.execute("ALTER TYPE image_status_enum_without_cancelled RENAME TO image_status_enum")
    op.execute(
        "ALTER TABLE wrong_images ALTER COLUMN status "
        "SET DEFAULT 'pending'::image_status_enum"
    )
