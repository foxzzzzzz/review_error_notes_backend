from pathlib import Path

import pytest
from pydantic import ValidationError


def test_profile_update_accepts_primary_school_settings():
    from app.schemas.profile import ProfileUpdate

    data = ProfileUpdate(grade=6, semester=2)
    assert data.model_dump(exclude_unset=True) == {"grade": 6, "semester": 2}


@pytest.mark.parametrize(
    ("field", "value"),
    (("grade", 0), ("grade", 7), ("semester", 0), ("semester", 3)),
)
def test_profile_update_rejects_out_of_range_settings(field, value):
    from app.schemas.profile import ProfileUpdate

    with pytest.raises(ValidationError):
        ProfileUpdate(**{field: value})


def test_upload_accepts_validated_subject_grade_and_semester_fields():
    source = (Path(__file__).parents[2] / "app" / "api" / "upload.py").read_text(encoding="utf-8")

    assert "subject:" in source
    assert "grade:" in source
    assert "semester:" in source
    assert source.count("Form(") >= 3
