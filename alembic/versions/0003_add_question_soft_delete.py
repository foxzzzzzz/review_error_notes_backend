"""Add soft deletion support for wrong questions.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-23
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "wrong_questions",
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_wrong_questions_deleted_at", "wrong_questions", ["deleted_at"], unique=False)

    op.drop_constraint("sheet_items_wrong_question_id_fkey", "sheet_items", type_="foreignkey")
    op.alter_column(
        "sheet_items",
        "wrong_question_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )
    op.create_foreign_key(
        "fk_sheet_items_wrong_question_id",
        "sheet_items",
        "wrong_questions",
        ["wrong_question_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.drop_constraint("sheet_items_derived_from_fkey", "sheet_items", type_="foreignkey")
    op.create_foreign_key(
        "fk_sheet_items_derived_from",
        "sheet_items",
        "wrong_questions",
        ["derived_from"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_sheet_items_derived_from", "sheet_items", type_="foreignkey")
    op.create_foreign_key(
        "sheet_items_derived_from_fkey",
        "sheet_items",
        "wrong_questions",
        ["derived_from"],
        ["id"],
    )

    op.drop_constraint("fk_sheet_items_wrong_question_id", "sheet_items", type_="foreignkey")
    # If physical cleanup has produced NULL source references, restoring the
    # original NOT NULL constraint must fail instead of deleting sheet history.
    op.alter_column(
        "sheet_items",
        "wrong_question_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.create_foreign_key(
        "sheet_items_wrong_question_id_fkey",
        "sheet_items",
        "wrong_questions",
        ["wrong_question_id"],
        ["id"],
    )

    op.drop_index("ix_wrong_questions_deleted_at", table_name="wrong_questions")
    op.drop_column("wrong_questions", "deleted_at")
