from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError


def test_profile_update_accepts_primary_school_settings():
    from app.schemas.profile import ProfileUpdate

    data = ProfileUpdate(
        nickname="  小树  ",
        student_name="  小树同学  ",
        grade=6,
        semester=2,
    )
    assert data.model_dump(exclude_unset=True) == {
        "nickname": "小树",
        "student_name": "小树同学",
        "grade": 6,
        "semester": 2,
    }


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


@pytest.mark.parametrize("field", ("nickname", "student_name"))
def test_profile_update_rejects_blank_names(field):
    from app.schemas.profile import ProfileUpdate

    with pytest.raises(ValidationError):
        ProfileUpdate(**{field: "   "})


def test_profile_response_exposes_account_student_prompt_and_real_stats():
    from app.schemas.profile import ProfileOut

    profile = ProfileOut(
        nickname="小树",
        avatar_url="/avatars/account.jpg",
        profile_prompt_required=False,
        student_id=UUID("00000000-0000-0000-0000-000000000002"),
        student_name="小树同学",
        grade=2,
        semester=1,
        student_profile_required=False,
        phone_bound=False,
        phone_masked="",
        stats={
            "total": 12,
            "month_new": 4,
            "needs_review": 2,
            "mastered": 3,
        },
    )

    assert profile.stats.total == 12
    assert profile.stats.month_new == 4
    assert profile.stats.needs_review == 2
    assert profile.stats.mastered == 3
