"""Create the final test-stage wrong-book schema.

Revision ID: 0001
Revises:
Create Date: 2026-07-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


subject_enum = postgresql.ENUM(
    "math", "chinese", "english", name="subject_enum", create_type=False
)
image_status_enum = postgresql.ENUM(
    "pending",
    "segmented",
    "needs_review",
    "confirmed",
    name="image_status_enum",
    create_type=False,
)
question_review_status_enum = postgresql.ENUM(
    "pending",
    "processing",
    "needs_review",
    "confirmed",
    name="question_review_status_enum",
    create_type=False,
)
question_mastery_status_enum = postgresql.ENUM(
    "learning",
    "mastered",
    name="question_mastery_status_enum",
    create_type=False,
)
item_type_enum = postgresql.ENUM(
    "original", "derived", name="item_type_enum", create_type=False
)


def _timestamps() -> tuple[sa.Column, sa.Column]:
    return (
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def upgrade() -> None:
    bind = op.get_bind()
    subject_enum.create(bind, checkfirst=True)
    image_status_enum.create(bind, checkfirst=True)
    question_review_status_enum.create(bind, checkfirst=True)
    question_mastery_status_enum.create(bind, checkfirst=True)
    item_type_enum.create(bind, checkfirst=True)

    op.create_table(
        "accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("nickname", sa.String(length=50), nullable=True),
        sa.Column("avatar_object_key", sa.String(length=255), nullable=True),
        sa.Column("phone_ciphertext", sa.String(length=512), nullable=True),
        sa.Column("phone_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("phone_bound_at", sa.DateTime(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="active",
            nullable=False,
        ),
        sa.Column("profile_prompted_at", sa.DateTime(), nullable=True),
        sa.Column("profile_completed_at", sa.DateTime(), nullable=True),
        sa.Column(
            "token_version",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
        sa.Column("deletion_requested_at", sa.DateTime(), nullable=True),
        sa.Column("deletion_due_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('active', 'pending_deletion')",
            name="ck_accounts_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_accounts_phone_fingerprint",
        "accounts",
        ["phone_fingerprint"],
        unique=True,
    )
    op.create_index("ix_accounts_status", "accounts", ["status"], unique=False)
    op.create_index(
        "ix_accounts_deletion_due_at",
        "accounts",
        ["deletion_due_at"],
        unique=False,
    )

    op.create_table(
        "wechat_identities",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("account_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("appid", sa.String(length=64), nullable=False),
        sa.Column("openid", sa.String(length=128), nullable=False),
        sa.Column("unionid", sa.String(length=128), nullable=True),
        sa.Column("last_login_at", sa.DateTime(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["accounts.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "appid",
            "openid",
            name="uq_wechat_identity_appid_openid",
        ),
    )
    op.create_index(
        "ix_wechat_identities_account_id",
        "wechat_identities",
        ["account_id"],
        unique=False,
    )

    op.create_table(
        "students",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("account_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("display_name", sa.String(length=50), nullable=True),
        sa.Column("grade", sa.SmallInteger(), nullable=True),
        sa.Column("semester", sa.SmallInteger(), nullable=True),
        sa.Column(
            "is_default",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
        sa.Column(
            "profile_completed",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["accounts.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_students_account_id",
        "students",
        ["account_id"],
        unique=False,
    )
    op.create_index(
        "uq_students_default_per_account",
        "students",
        ["account_id"],
        unique=True,
        postgresql_where=sa.text("is_default"),
    )

    op.create_table(
        "wrong_images",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("original_url", sa.String(length=500), nullable=False),
        sa.Column("subject", subject_enum, nullable=True),
        sa.Column("grade", sa.SmallInteger(), nullable=False),
        sa.Column("semester", sa.SmallInteger(), nullable=False),
        sa.Column("question_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", image_status_enum, server_default="pending", nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["student_id"], ["students.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_wrong_images_student_id",
        "wrong_images",
        ["student_id"],
        unique=False,
    )

    op.create_table(
        "wrong_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("image_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("crop_region", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("subject", subject_enum, nullable=True),
        sa.Column("semester", sa.SmallInteger(), nullable=False),
        sa.Column("grade", sa.SmallInteger(), nullable=False),
        sa.Column("ocr_text", sa.Text(), nullable=True),
        sa.Column("ocr_answer", sa.Text(), nullable=True),
        sa.Column("ocr_raw_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("question_type", sa.String(length=20), nullable=True),
        sa.Column("problem_schema", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("difficulty_params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("derived_schema", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String()), server_default="{}", nullable=False),
        sa.Column("difficulty", sa.SmallInteger(), nullable=True),
        sa.Column("wrong_count", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "review_status",
            question_review_status_enum,
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "mastery_status",
            question_mastery_status_enum,
            server_default="learning",
            nullable=False,
        ),
        sa.Column("mastered_at", sa.DateTime(), nullable=True),
        sa.Column("last_practiced_at", sa.DateTime(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["image_id"], ["wrong_images.id"]),
        sa.ForeignKeyConstraint(["student_id"], ["students.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_wrong_questions_image_id",
        "wrong_questions",
        ["image_id"],
        unique=False,
    )
    op.create_index(
        "ix_wrong_questions_student_id",
        "wrong_questions",
        ["student_id"],
        unique=False,
    )
    op.create_index(
        "ix_wrong_questions_deleted_at",
        "wrong_questions",
        ["deleted_at"],
        unique=False,
    )

    op.create_table(
        "practice_sheets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=100), nullable=True),
        sa.Column("config_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("pdf_url", sa.String(length=500), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["student_id"], ["students.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_practice_sheets_student_id",
        "practice_sheets",
        ["student_id"],
        unique=False,
    )

    op.create_table(
        "sheet_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sheet_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("wrong_question_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("question_type", item_type_enum, nullable=False),
        sa.Column("derived_from", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("sort_order", sa.Integer(), server_default="0", nullable=False),
        sa.Column("generation_method", sa.String(length=20), nullable=True),
        sa.ForeignKeyConstraint(
            ["derived_from"],
            ["wrong_questions.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(["sheet_id"], ["practice_sheets.id"]),
        sa.ForeignKeyConstraint(
            ["wrong_question_id"],
            ["wrong_questions.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_sheet_items_sheet_id",
        "sheet_items",
        ["sheet_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_sheet_items_sheet_id", table_name="sheet_items")
    op.drop_table("sheet_items")
    op.drop_index(
        "ix_practice_sheets_student_id",
        table_name="practice_sheets",
    )
    op.drop_table("practice_sheets")
    op.drop_index("ix_wrong_questions_deleted_at", table_name="wrong_questions")
    op.drop_index("ix_wrong_questions_student_id", table_name="wrong_questions")
    op.drop_index("ix_wrong_questions_image_id", table_name="wrong_questions")
    op.drop_table("wrong_questions")
    op.drop_index("ix_wrong_images_student_id", table_name="wrong_images")
    op.drop_table("wrong_images")
    op.drop_index("uq_students_default_per_account", table_name="students")
    op.drop_index("ix_students_account_id", table_name="students")
    op.drop_table("students")
    op.drop_index(
        "ix_wechat_identities_account_id",
        table_name="wechat_identities",
    )
    op.drop_table("wechat_identities")
    op.drop_index("ix_accounts_deletion_due_at", table_name="accounts")
    op.drop_index("ix_accounts_status", table_name="accounts")
    op.drop_index("ix_accounts_phone_fingerprint", table_name="accounts")
    op.drop_table("accounts")

    bind = op.get_bind()
    item_type_enum.drop(bind, checkfirst=True)
    question_mastery_status_enum.drop(bind, checkfirst=True)
    question_review_status_enum.drop(bind, checkfirst=True)
    image_status_enum.drop(bind, checkfirst=True)
    subject_enum.drop(bind, checkfirst=True)
