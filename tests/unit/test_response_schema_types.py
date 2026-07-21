from pathlib import Path


SCHEMA_DIR = Path(__file__).parents[2] / "app" / "schemas"


def test_orm_uuid_fields_use_uuid_types_for_pydantic_validation():
    question = (SCHEMA_DIR / "question.py").read_text(encoding="utf-8")
    sheet = (SCHEMA_DIR / "sheet.py").read_text(encoding="utf-8")

    assert "id: UUID" in question
    assert "image_id: UUID" in question
    assert "id: UUID" in sheet
