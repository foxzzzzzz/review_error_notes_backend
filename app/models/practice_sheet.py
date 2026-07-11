from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.models import Base, TimestampMixin, gen_uuid


class PracticeSheet(Base, TimestampMixin):
    __tablename__ = "practice_sheets"
    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id"), nullable=False, index=True)
    title = Column(String(100), nullable=True)
    config_json = Column(JSONB, nullable=True)
    pdf_url = Column(String(500), nullable=True)
