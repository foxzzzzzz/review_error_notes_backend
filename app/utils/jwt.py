from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError
from app.config import settings


NORMAL_TOKEN_SCOPE = "normal"
ACCOUNT_RECOVERY_TOKEN_SCOPE = "account_recovery"


@dataclass(frozen=True)
class TokenClaims:
    account_id: str
    token_version: int
    scope: str = NORMAL_TOKEN_SCOPE


def create_token(account_id: str, token_version: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_EXPIRE_DAYS)
    payload = {
        "sub": str(account_id),
        "ver": token_version,
        "scope": NORMAL_TOKEN_SCOPE,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_account_recovery_token(
    account_id: str,
    token_version: int,
    expires_at: datetime,
) -> str:
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    payload = {
        "sub": str(account_id),
        "ver": token_version,
        "scope": ACCOUNT_RECOVERY_TOKEN_SCOPE,
        "exp": expires_at,
    }
    return jwt.encode(
        payload,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )


def verify_token(
    token: str,
    required_scope: str = NORMAL_TOKEN_SCOPE,
    allow_expired: bool = False,
) -> TokenClaims | None:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": not allow_expired},
        )
        account_id = payload.get("sub")
        token_version = payload.get("ver")
        scope = payload.get("scope")
        if (
            not account_id
            or not isinstance(token_version, int)
            or scope != required_scope
        ):
            return None
        return TokenClaims(
            account_id=str(account_id),
            token_version=token_version,
            scope=scope,
        )
    except JWTError:
        return None
