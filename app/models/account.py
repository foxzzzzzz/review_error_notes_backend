from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from app.models import Base, TimestampMixin, gen_uuid


class Account(Base, TimestampMixin):
    __tablename__ = "accounts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    nickname = Column(String(50), nullable=True)
    avatar_object_key = Column(String(255), nullable=True)
    phone_ciphertext = Column(String(512), nullable=True)
    phone_fingerprint = Column(String(64), nullable=True, unique=True, index=True)
    phone_bound_at = Column(DateTime, nullable=True)
    status = Column(String(32), nullable=False, default="active", index=True)
    profile_prompted_at = Column(DateTime, nullable=True)
    profile_completed_at = Column(DateTime, nullable=True)
    token_version = Column(Integer, nullable=False, default=1)
    deletion_requested_at = Column(DateTime, nullable=True)
    deletion_due_at = Column(DateTime, nullable=True, index=True)
