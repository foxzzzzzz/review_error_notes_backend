from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProfileStats(BaseModel):
    total: int = 0
    month_new: int = 0
    needs_review: int = 0
    mastered: int = 0


class ProfileOut(BaseModel):
    nickname: Optional[str]
    avatar_url: Optional[str]
    profile_prompt_required: bool
    student_id: UUID
    student_name: Optional[str]
    grade: Optional[int]
    semester: Optional[int]
    student_profile_required: bool
    phone_bound: bool
    phone_masked: str = ""
    stats: ProfileStats

    model_config = ConfigDict(from_attributes=True)


class ProfileUpdate(BaseModel):
    nickname: Optional[str] = Field(default=None, max_length=50)
    student_name: Optional[str] = Field(default=None, max_length=50)
    grade: Optional[int] = Field(default=None, ge=1, le=6)
    semester: Optional[int] = Field(default=None, ge=1, le=2)

    @field_validator("nickname", "student_name")
    @classmethod
    def names_must_not_be_blank(cls, value):
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value
