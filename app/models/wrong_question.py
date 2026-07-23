from sqlalchemy import Column, DateTime, String, SmallInteger, Integer, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM, ARRAY
from app.models import Base, TimestampMixin, gen_uuid


class WrongQuestion(Base, TimestampMixin):
    __tablename__ = "wrong_questions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id"), nullable=False, index=True)
    image_id = Column(UUID(as_uuid=True), ForeignKey("wrong_images.id"), nullable=False, index=True)
    crop_region = Column(JSONB, nullable=True)
    subject = Column(ENUM("math", "chinese", "english", name="subject_enum"), nullable=True)
    semester = Column(SmallInteger, nullable=False)
    grade = Column(SmallInteger, nullable=False)
    ocr_text = Column(Text, nullable=True)
    ocr_answer = Column(Text, nullable=True)
    ocr_raw_json = Column(JSONB, nullable=True)
    question_type = Column(String(20), nullable=True)
    problem_schema = Column(JSONB, nullable=True)
    difficulty_params = Column(JSONB, nullable=True)
    derived_schema = Column(JSONB, nullable=True)
    tags = Column(ARRAY(String), default=[])
    difficulty = Column(SmallInteger, nullable=True)
    wrong_count = Column(Integer, default=1)
    deleted_at = Column(DateTime, nullable=True, index=True)
    status = Column(
        ENUM("pending", "ocr_done", "needs_review", "confirmed", "mastered", name="question_status_enum"),
        default="pending"
    )
