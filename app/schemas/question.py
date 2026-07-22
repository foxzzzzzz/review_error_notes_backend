from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID

class QuestionOut(BaseModel):
    id: UUID
    image_id: UUID
    subject: Optional[str]
    grade: int
    semester: int
    ocr_text: Optional[str]
    ocr_answer: Optional[str]
    ocr_raw_json: Optional[dict]
    crop_region: Optional[dict] = None
    image_url: Optional[str] = None
    question_type: Optional[str]
    tags: list[str]
    difficulty: Optional[int]
    wrong_count: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class QuestionUpdate(BaseModel):
    subject: Optional[str] = None
    ocr_text: Optional[str] = None
    question_type: Optional[str] = None
    tags: Optional[list[str]] = None
    difficulty: Optional[int] = None
    status: Optional[str] = None
