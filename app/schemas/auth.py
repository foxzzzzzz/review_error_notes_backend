from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    code: str


class LoginResponse(BaseModel):
    token: str
    need_phone: bool
    student_id: str


class BindPhoneRequest(BaseModel):
    code: str = Field(min_length=1)
