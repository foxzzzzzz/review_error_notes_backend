from sqlalchemy import Column, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID

from app.models import Base, TimestampMixin, gen_uuid


class WeChatIdentity(Base, TimestampMixin):
    __tablename__ = "wechat_identities"
    __table_args__ = (
        UniqueConstraint(
            "appid",
            "openid",
            name="uq_wechat_identity_appid_openid",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    account_id = Column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    appid = Column(String(64), nullable=False)
    openid = Column(String(128), nullable=False)
    unionid = Column(String(128), nullable=True)
    last_login_at = Column(DateTime, nullable=False)
