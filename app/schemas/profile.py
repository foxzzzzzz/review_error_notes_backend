from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ProfileOut(BaseModel):
    nickname: Optional[str]
    avatar_url: Optional[str]
    grade: int
    semester: int
    phone_bound: bool
    phone_masked: str = ""

    model_config = ConfigDict(from_attributes=True)


class ProfileUpdate(BaseModel):
    grade: Optional[int] = Field(default=None, ge=1, le=6)
    semester: Optional[int] = Field(default=None, ge=1, le=2)
