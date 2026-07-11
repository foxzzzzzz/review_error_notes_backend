from sqlalchemy import Column, String, SmallInteger, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ENUM
from app.models import Base, TimestampMixin, gen_uuid


class WrongImage(Base, TimestampMixin):
    __tablename__ = "wrong_images"
    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id"), nullable=False, index=True)
    original_url = Column(String(500), nullable=False)
    subject = Column(ENUM("math", "chinese", "english", name="subject_enum"), nullable=True)
    grade = Column(SmallInteger, nullable=False)
    semester = Column(SmallInteger, nullable=False)
    question_count = Column(Integer, default=0)
    status = Column(ENUM("pending", "segmented", "confirmed", name="image_status_enum"), default="pending")
