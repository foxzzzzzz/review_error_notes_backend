from sqlalchemy import Boolean, Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.models import Base, TimestampMixin, gen_uuid


class PracticeResult(Base, TimestampMixin):
    __tablename__ = "practice_results"
    __table_args__ = (
        UniqueConstraint(
            "attempt_id",
            "sheet_item_id",
            name="uq_result_attempt_item",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    attempt_id = Column(
        UUID(as_uuid=True),
        ForeignKey("practice_attempts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sheet_item_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sheet_items.id"),
        nullable=False,
    )
    wrong_question_id = Column(
        UUID(as_uuid=True),
        ForeignKey("wrong_questions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    is_correct = Column(Boolean, nullable=False)
    derived_count = Column(Integer, nullable=False, default=0)
    question_snapshot = Column(JSONB, nullable=False)
