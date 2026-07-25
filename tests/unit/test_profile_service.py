import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from app.schemas.profile import ProfileStats, ProfileUpdate


class Result:
    def one(self):
        return (12, 4, 2, 3)


class StatsDB:
    def __init__(self):
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return Result()


def test_beijing_month_start_is_converted_to_utc_naive():
    from app.services.profile import beijing_month_start_utc_naive

    now = datetime(2026, 8, 1, 0, 30, tzinfo=timezone.utc)

    assert beijing_month_start_utc_naive(now) == datetime(2026, 7, 31, 16, 0)


def test_profile_statistics_count_only_visible_current_student_questions():
    from app.services.profile import load_profile_stats

    db = StatsDB()
    stats = asyncio.run(
        load_profile_stats(
            db,
            student_id="student-1",
            now=datetime(2026, 8, 1, 0, 30, tzinfo=timezone.utc),
        )
    )
    sql = str(db.statement)

    assert stats.model_dump() == {
        "total": 12,
        "month_new": 4,
        "needs_review": 2,
        "mastered": 3,
    }
    assert "wrong_questions.student_id" in sql
    assert "wrong_questions.deleted_at IS NULL" in sql
    assert "wrong_questions.created_at >=" in sql
    assert "wrong_questions.review_status" in sql
    assert "wrong_questions.mastery_status" in sql


def test_profile_update_marks_account_and_student_completion():
    from app.schemas.profile import ProfileUpdate
    from app.services.profile import apply_profile_update

    account = SimpleNamespace(
        nickname=None,
        profile_prompted_at=None,
        profile_completed_at=None,
    )
    student = SimpleNamespace(
        display_name=None,
        grade=None,
        semester=None,
        profile_completed=False,
    )
    now = datetime(2026, 7, 25, 8, 0)

    apply_profile_update(
        account,
        student,
        ProfileUpdate(
            nickname="小树",
            student_name="小树同学",
            grade=2,
            semester=1,
        ),
        now=now,
    )

    assert account.nickname == "小树"
    assert account.profile_prompted_at == now
    assert account.profile_completed_at == now
    assert student.display_name == "小树同学"
    assert student.grade == 2
    assert student.semester == 1
    assert student.profile_completed is True


def test_skipping_profile_prompt_is_idempotent():
    from app.services.profile import mark_profile_prompt_skipped

    original = datetime(2026, 7, 25, 8, 0)
    account = SimpleNamespace(profile_prompted_at=original)

    mark_profile_prompt_skipped(
        account,
        now=datetime(2026, 7, 25, 9, 0),
    )

    assert account.profile_prompted_at == original


def test_profile_update_reloads_and_locks_student_before_computing_completion(
    monkeypatch,
):
    from app.api import profile as profile_api

    account_id = uuid4()
    student_id = uuid4()
    account = SimpleNamespace(
        id=account_id,
        nickname=None,
        avatar_object_key=None,
        phone_ciphertext=None,
        profile_prompted_at=None,
        profile_completed_at=None,
    )
    dependency_student = SimpleNamespace(
        id=student_id,
        account_id=account_id,
        display_name=None,
        grade=None,
        semester=None,
        profile_completed=False,
    )
    locked_student = SimpleNamespace(
        id=student_id,
        account_id=account_id,
        display_name=None,
        grade=None,
        semester=1,
        profile_completed=False,
    )

    class ProfileDB:
        def __init__(self):
            self.scalar_calls = []
            self.results = [account, locked_student]

        async def scalar(self, statement):
            self.scalar_calls.append(statement)
            return self.results.pop(0)

        async def commit(self):
            pass

        async def refresh(self, _instance):
            pass

    async def load_stats(_db, _student_id):
        return ProfileStats()

    monkeypatch.setattr(profile_api, "load_profile_stats", load_stats)
    db = ProfileDB()

    result = asyncio.run(
        profile_api.update_profile(
            ProfileUpdate(grade=1),
            student=dependency_student,
            db=db,
        )
    )

    assert len(db.scalar_calls) == 2
    assert all("FOR UPDATE" in str(statement) for statement in db.scalar_calls)
    assert "students" in str(db.scalar_calls[1])
    assert locked_student.grade == 1
    assert locked_student.semester == 1
    assert locked_student.profile_completed is True
    assert result.student_profile_required is False
