from pydantic import BaseModel, Field, model_validator
from typing import Optional
from datetime import datetime
from uuid import UUID


class SheetCreate(BaseModel):
    title: str = "错题重练"
    question_ids: list[str]
    derived_per_original: int = Field(default=0, ge=0, le=3)
    difficulty_boost: int = Field(default=2, ge=1, le=3)


class SheetItemOut(BaseModel):
    id: UUID
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
    updated_at: datetime
    generation_status: str = "completed"
    generation_total: int = 0
    generation_completed: int = 0
    generation_error_code: Optional[str] = None
    generation_error_message: Optional[str] = None
    generation_duration_seconds: Optional[int] = None
    items: list[SheetItemOut] = []
    latest_accuracy: Optional[float] = None
    attempt_count: int = 0
    practice_status: str = "unpracticed"

    class Config:
        from_attributes = True


class SheetGenerationOut(BaseModel):
    id: UUID
    generation_status: str
    generation_total: int
    generation_completed: int
    generation_error_code: Optional[str] = None
    generation_error_message: Optional[str] = None
    generation_duration_seconds: Optional[int] = None
    pdf_url: Optional[str]
    updated_at: datetime

    class Config:
        from_attributes = True


class AttemptItemInput(BaseModel):
    sheet_item_id: UUID
    is_correct: bool


class AttemptCreate(BaseModel):
    idempotency_key: str = Field(min_length=1, max_length=64)
    completed_at: datetime
    items: list[AttemptItemInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_items(self):
        item_ids = [item.sheet_item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("duplicate sheet item IDs")
        return self


class AttemptUpdate(BaseModel):
    updated_at: datetime
    items: list[AttemptItemInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_items(self):
        item_ids = [item.sheet_item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("duplicate sheet item IDs")
        return self


class AttemptResultOut(BaseModel):
    sheet_item_id: UUID
    is_correct: bool


class AttemptOut(BaseModel):
    id: UUID
    sheet_id: UUID
    attempt_no: int
    correct_count: int
    total_count: int
    accuracy: float
    completed_at: datetime
    updated_at: datetime
    items: list[AttemptResultOut]


class ReviewItemOut(BaseModel):
    sheet_item_id: UUID
    question_type: str
    question_text: str
    sort_order: int
    is_correct: bool


class ReviewGroupOut(BaseModel):
    wrong_question_id: Optional[UUID]
    items: list[ReviewItemOut]


class SheetReviewOut(BaseModel):
    sheet_id: UUID
    title: Optional[str]
    latest_attempt: Optional[AttemptOut]
    groups: list[ReviewGroupOut]
