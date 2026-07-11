from pydantic import BaseModel


class LoginRequest(BaseModel):
    code: str


class LoginResponse(BaseModel):
    token: str
    need_phone: bool
    student_id: str


class BindPhoneRequest(BaseModel):
    encrypted_data: str
    iv: str
