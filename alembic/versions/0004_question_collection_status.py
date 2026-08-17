"""Add collection state for recognized question candidates.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-17
"""

from alembic import op
import sqlalchemy as sa


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "wrong_questions",
        sa.Column(
            "collection_status",
            sa.String(length=32),
            nullable=False,
            server_default="collected",
        ),
    )
    op.create_index(
        "ix_wrong_questions_collection_status",
        "wrong_questions",
        ["collection_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_wrong_questions_collection_status", table_name="wrong_questions")
    op.drop_column("wrong_questions", "collection_status")
