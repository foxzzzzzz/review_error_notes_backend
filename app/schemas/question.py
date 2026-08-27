from pydantic import BaseModel, Field
from typing import Literal, Optional
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
    question_type: Optional[str]
    tags: list[str]
    difficulty: Optional[int]
    wrong_count: int
    collection_status: str
    review_status: str
    mastery_status: str
    created_at: datetime

    class Config:
        from_attributes = True

class QuestionUpdate(BaseModel):
    subject: Optional[str] = None
    ocr_text: Optional[str] = None
    question_type: Optional[str] = None
    tags: Optional[list[str]] = None
    difficulty: Optional[int] = None
    review_status: Optional[str] = None


class ReviewDecision(BaseModel):
    question_id: UUID
    decision: Literal["collect", "ignore"]


class ReviewDecisionRequest(BaseModel):
    decisions: list[ReviewDecision] = Field(min_length=1)


class ReviewImageReprocessRequest(BaseModel):
    correction: Literal[
        "missed_errors",
        "false_positives",
        "both",
        "force_unmarked",
    ]
