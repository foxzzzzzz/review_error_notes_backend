"""add image-level recognition audit payload

Revision ID: 0010
Revises: 0009
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "wrong_images",
        sa.Column("recognition_audit_json", postgresql.JSONB(astext_type=None), nullable=True),
    )


def downgrade():
    op.drop_column("wrong_images", "recognition_audit_json")
