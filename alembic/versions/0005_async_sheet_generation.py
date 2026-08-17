"""Add asynchronous practice-sheet generation state.

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-17
"""

from alembic import op
import sqlalchemy as sa


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "practice_sheets",
        sa.Column(
            "generation_status",
            sa.String(length=32),
            nullable=False,
            server_default="completed",
        ),
    )
    op.add_column(
        "practice_sheets",
        sa.Column(
            "generation_total",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "practice_sheets",
        sa.Column(
            "generation_completed",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "practice_sheets",
        sa.Column("generation_error_code", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "practice_sheets",
        sa.Column("generation_error_message", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("practice_sheets", "generation_error_message")
    op.drop_column("practice_sheets", "generation_error_code")
    op.drop_column("practice_sheets", "generation_completed")
    op.drop_column("practice_sheets", "generation_total")
    op.drop_column("practice_sheets", "generation_status")
