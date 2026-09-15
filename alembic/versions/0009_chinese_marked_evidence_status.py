"""Add persisted Chinese marked-evidence pipeline statuses.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-15
"""

from typing import Union

from alembic import op
import sqlalchemy as sa


revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column("wrong_questions", sa.Column("recognition_pipeline", sa.String(length=64), nullable=True))
    op.add_column("wrong_questions", sa.Column("mark_status", sa.String(length=32), nullable=True))
    op.add_column("wrong_questions", sa.Column("question_evidence_status", sa.String(length=32), nullable=True))
    op.add_column("wrong_questions", sa.Column("answer_status", sa.String(length=32), nullable=True))
    op.create_index(
        "ix_wrong_questions_recognition_pipeline",
        "wrong_questions",
        ["recognition_pipeline"],
    )
    op.create_index(
        "ix_wrong_questions_answer_status",
        "wrong_questions",
        ["answer_status"],
    )


def downgrade() -> None:
    op.drop_index("ix_wrong_questions_answer_status", table_name="wrong_questions")
    op.drop_index("ix_wrong_questions_recognition_pipeline", table_name="wrong_questions")
    op.drop_column("wrong_questions", "answer_status")
    op.drop_column("wrong_questions", "question_evidence_status")
    op.drop_column("wrong_questions", "mark_status")
    op.drop_column("wrong_questions", "recognition_pipeline")
