"""Add one-time image recognition correction context.

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-20
"""

from typing import Union

from alembic import op
import sqlalchemy as sa


revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column(
        "wrong_images",
        sa.Column("recognition_correction", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("wrong_images", "recognition_correction")
