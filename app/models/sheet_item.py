from sqlalchemy import Column, String, Integer, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ENUM
from app.models import Base, gen_uuid


class SheetItem(Base):
    __tablename__ = "sheet_items"
    id = Column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    sheet_id = Column(UUID(as_uuid=True), ForeignKey("practice_sheets.id"), nullable=False, index=True)
    wrong_question_id = Column(
        UUID(as_uuid=True), ForeignKey("wrong_questions.id", ondelete="SET NULL"), nullable=True
    )
    question_type = Column(ENUM("original", "derived", name="item_type_enum"), nullable=False)
    derived_from = Column(
        UUID(as_uuid=True), ForeignKey("wrong_questions.id", ondelete="SET NULL"), nullable=True
    )
    question_text = Column(Text, nullable=False)
    sort_order = Column(Integer, default=0)
    generation_method = Column(String(20), nullable=True)
