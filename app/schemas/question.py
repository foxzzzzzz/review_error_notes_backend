from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class QuestionOut(BaseModel):
    id: str
    image_id: str
    subject: Optional[str]
    grade: int
    semester: int
    ocr_text: Optional[str]
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
