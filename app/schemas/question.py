from pydantic import BaseModel, Field, field_validator
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
    recognition_pipeline: Optional[str] = None
    mark_status: Optional[str] = None
    question_evidence_status: Optional[str] = None
    answer_status: Optional[str] = None
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
    correct_answer: str | None = Field(default=None, max_length=1000)
    question_type: str | None = Field(default=None, max_length=20)
    instruction: str | None = Field(default=None, max_length=500)
    prompt_text: str | None = Field(default=None, max_length=5000)


class ReviewDecisionRequest(BaseModel):
    decisions: list[ReviewDecision] = Field(min_length=1)


class ReviewImageReprocessRequest(BaseModel):
    correction: Literal[
        "missed_errors",
        "false_positives",
        "both",
        "force_unmarked",
    ]


class ManualWrongQuestionRequest(BaseModel):
    question_id: UUID
    bbox: list[float] = Field(min_length=4, max_length=4)
    instruction: str = Field(min_length=1, max_length=500)
    prompt_text: str = Field(min_length=1, max_length=5000)
    question_type: str = Field(min_length=1, max_length=20)
    correct_answer: str = Field(min_length=1, max_length=1000)
    student_answer: str | None = Field(default=None, max_length=1000)

    @field_validator("bbox", mode="before")
    @classmethod
    def reject_boolean_bbox_coordinates(cls, value):
        if isinstance(value, (list, tuple)) and any(
            isinstance(coordinate, bool) for coordinate in value
        ):
            raise ValueError("bbox coordinates must be numbers")
        return value


class ManualQuestionSuggestionRequest(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    mode: Literal["ocr", "llm"]

    @field_validator("bbox", mode="before")
    @classmethod
    def reject_boolean_bbox_coordinates(cls, value):
        if isinstance(value, (list, tuple)) and any(
            isinstance(coordinate, bool) for coordinate in value
        ):
            raise ValueError("bbox coordinates must be numbers")
        return value
