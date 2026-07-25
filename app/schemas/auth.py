from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    code: str


class LoginResponse(BaseModel):
    token: str
    account_id: UUID
    student_id: UUID
    is_new_account: bool
    profile_prompt_required: bool
    student_profile_required: bool
    account_status: Literal["active", "pending_deletion"]


class BindPhoneRequest(BaseModel):
    code: str = Field(min_length=1)


class BindPhoneResponse(BaseModel):
    status: Literal["bound"]
    phone_masked: str


class RecoverAccountRequest(BaseModel):
    recovery_token: str = Field(min_length=1)
