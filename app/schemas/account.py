from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AccountDeletionRequest(BaseModel):
    code: str = Field(min_length=1)


class AccountDeletionResponse(BaseModel):
    account_status: Literal["pending_deletion"]
    deletion_due_at: datetime
    recovery_token: str


class OperationResponse(BaseModel):
    ok: bool
