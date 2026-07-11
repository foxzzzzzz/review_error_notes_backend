from sqlalchemy import Column, String, Boolean, SmallInteger
from sqlalchemy.dialects.postgresql import UUID
from app.models import Base, TimestampMixin, gen_uuid


class Student(Base, TimestampMixin):
    __tablename__ = "students"
    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    openid = Column(String(64), unique=True, nullable=False, index=True)
    phone = Column(String(64), nullable=True)
    phone_bound = Column(Boolean, default=False)
    nickname = Column(String(50), nullable=True)
    grade = Column(SmallInteger, nullable=False, default=1)
    semester = Column(SmallInteger, nullable=False, default=1)
    avatar_url = Column(String(255), nullable=True)
