"""Add practice-sheet generation timing.

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-18
"""

from typing import Union

from alembic import op
import sqlalchemy as sa


revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column(
        "practice_sheets",
        sa.Column("generation_started_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "practice_sheets",
        sa.Column("generation_duration_seconds", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("practice_sheets", "generation_duration_seconds")
    op.drop_column("practice_sheets", "generation_started_at")
