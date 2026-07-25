from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID

from app.models import Base, TimestampMixin, gen_uuid


class PracticeAttempt(Base, TimestampMixin):
    __tablename__ = "practice_attempts"
    __table_args__ = (
        UniqueConstraint(
            "student_id",
            "idempotency_key",
            name="uq_attempt_student_key",
        ),
        UniqueConstraint(
            "sheet_id",
            "attempt_no",
            name="uq_attempt_sheet_number",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    sheet_id = Column(
        UUID(as_uuid=True),
        ForeignKey("practice_sheets.id"),
        nullable=False,
        index=True,
    )
    student_id = Column(
        UUID(as_uuid=True),
        ForeignKey("students.id"),
        nullable=False,
        index=True,
    )
    attempt_no = Column(Integer, nullable=False)
    idempotency_key = Column(String(64), nullable=False)
    correct_count = Column(Integer, nullable=False)
    total_count = Column(Integer, nullable=False)
    accuracy = Column(Numeric(5, 4), nullable=False)
    completed_at = Column(DateTime, nullable=False)
