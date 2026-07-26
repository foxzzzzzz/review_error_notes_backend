from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    code: str


class LoginResponse(BaseModel):
    token: Optional[str] = None
    recovery_token: Optional[str] = None
    account_id: UUID
    student_id: UUID
    is_new_account: bool
    profile_prompt_required: bool
    student_profile_required: bool
    account_status: Literal["active", "pending_deletion"]
    deletion_due_at: Optional[datetime] = None


class BindPhoneRequest(BaseModel):
    code: str = Field(min_length=1)


class BindPhoneResponse(BaseModel):
    status: Literal["bound"]
    phone_masked: str


class RecoverAccountRequest(BaseModel):
    recovery_token: str = Field(min_length=1)
