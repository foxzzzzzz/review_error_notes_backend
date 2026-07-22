"""Add review statuses used by multimodal image recognition.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-22
"""

from typing import Sequence, Union

from alembic import op


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE question_status_enum ADD VALUE IF NOT EXISTS 'needs_review'")
    op.execute("ALTER TYPE image_status_enum ADD VALUE IF NOT EXISTS 'needs_review'")


def downgrade() -> None:
    # PostgreSQL cannot safely remove an enum value while rows may still use it.
    pass
