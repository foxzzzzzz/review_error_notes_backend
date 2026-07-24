from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError
from app.config import settings


@dataclass(frozen=True)
class TokenClaims:
    account_id: str
    token_version: int


def create_token(account_id: str, token_version: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_EXPIRE_DAYS)
    payload = {
        "sub": str(account_id),
        "ver": token_version,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def verify_token(token: str) -> TokenClaims | None:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        account_id = payload.get("sub")
        token_version = payload.get("ver")
        if not account_id or not isinstance(token_version, int):
            return None
        return TokenClaims(
            account_id=str(account_id),
            token_version=token_version,
        )
    except JWTError:
        return None
