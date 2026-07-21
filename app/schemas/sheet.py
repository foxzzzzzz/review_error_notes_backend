from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SheetCreate(BaseModel):
    title: str = "错题重练"
    question_ids: list[str]
    derived_per_original: int = Field(default=1, ge=1, le=3)
    difficulty_boost: int = Field(default=2, ge=1, le=3)


class SheetItemOut(BaseModel):
    question_type: str
    question_text: str
    sort_order: int

    class Config:
        from_attributes = True


class SheetOut(BaseModel):
    id: str
    title: Optional[str]
    config_json: Optional[dict]
    pdf_url: Optional[str]
    created_at: datetime
    items: list[SheetItemOut] = []

    class Config:
        from_attributes = True
