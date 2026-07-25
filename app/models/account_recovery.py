from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from app.models import Base, TimestampMixin, gen_uuid


class AccountRecoveryConflict(Base, TimestampMixin):
    __tablename__ = "account_recovery_conflicts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    support_reference = Column(
        String(15),
        nullable=False,
        unique=True,
        index=True,
    )
    current_account_id = Column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    target_account_id = Column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status = Column(String(32), nullable=False, default="open", index=True)
    resolved_at = Column(DateTime, nullable=True)


class FileCleanupJob(Base, TimestampMixin):
    __tablename__ = "file_cleanup_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    storage_kind = Column(String(32), nullable=False)
    object_path = Column(String(500), nullable=False)
    attempt_count = Column(Integer, nullable=False, default=0)
    last_error = Column(String(255), nullable=True)
    next_attempt_at = Column(DateTime, nullable=True, index=True)
