from sqlalchemy import Boolean, Column, ForeignKey, SmallInteger, String
from sqlalchemy.dialects.postgresql import UUID
from app.models import Base, TimestampMixin, gen_uuid


class Student(Base, TimestampMixin):
    __tablename__ = "students"

    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    account_id = Column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    display_name = Column(String(50), nullable=True)
    grade = Column(SmallInteger, nullable=True)
    semester = Column(SmallInteger, nullable=True)
    is_default = Column(Boolean, nullable=False, default=True)
    profile_completed = Column(Boolean, nullable=False, default=False)
