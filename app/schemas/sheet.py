from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


class SheetCreate(BaseModel):
    title: str = "错题重练"
    question_ids: list[str]
    derived_per_original: int = Field(default=0, ge=0, le=3)
    difficulty_boost: int = Field(default=2, ge=1, le=3)


class SheetItemOut(BaseModel):
    question_type: str
    question_text: str
    sort_order: int

    class Config:
        from_attributes = True


class SheetOut(BaseModel):
    id: UUID
    title: Optional[str]
    config_json: Optional[dict]
    pdf_url: Optional[str]
    created_at: datetime
    items: list[SheetItemOut] = []

    class Config:
        from_attributes = True
