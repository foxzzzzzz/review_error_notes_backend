import asyncio
import importlib.util
import logging
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException


MIGRATION_PATH = (
    Path(__file__).parents[2]
    / "alembic"
    / "versions"
    / "0005_async_sheet_generation.py"
)
DURATION_MIGRATION_PATH = (
    Path(__file__).parents[2]
    / "alembic"
    / "versions"
    / "0006_sheet_duration.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("async_sheet_migration", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_duration_migration():
    spec = importlib.util.spec_from_file_location(
        "sheet_duration_migration",
        DURATION_MIGRATION_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _MigrationRecorder:
    def __init__(self):
        self.added = []
        self.dropped = []

    def add_column(self, table_name, column):
        self.added.append((table_name, column))

    def drop_column(self, table_name, column_name):
        self.dropped.append((table_name, column_name))


def test_async_sheet_migration_preserves_existing_sheets_as_completed():
    migration = _load_migration()
    recorder = _MigrationRecorder()
    migration.op = recorder

    migration.upgrade()

    assert migration.down_revision == "0004"
    assert [column.name for _, column in recorder.added] == [
        "generation_status",
        "generation_total",
        "generation_completed",
        "generation_error_code",
        "generation_error_message",
    ]
    status_column = recorder.added[0][1]
    assert status_column.nullable is False
    assert str(status_column.server_default.arg) == "completed"

    migration.downgrade()
    assert recorder.dropped == [
        ("practice_sheets", "generation_error_message"),
        ("practice_sheets", "generation_error_code"),
        ("practice_sheets", "generation_completed"),
        ("practice_sheets", "generation_total"),
        ("practice_sheets", "generation_status"),
    ]


def test_practice_sheet_model_exposes_generation_state_columns():
    from app.models.practice_sheet import PracticeSheet

    columns = PracticeSheet.__table__.c
    assert str(columns.generation_status.default.arg) == "completed"
    assert columns.generation_total.default.arg == 0
    assert columns.generation_completed.default.arg == 0
    assert columns.generation_error_code.nullable is True
    assert columns.generation_error_message.nullable is True


def test_sheet_duration_migration_adds_nullable_timing_columns():
    migration = _load_duration_migration()
    recorder = _MigrationRecorder()
    migration.op = recorder

    migration.upgrade()

    assert migration.down_revision == "0005"
    assert [column.name for _, column in recorder.added] == [
        "generation_started_at",
        "generation_duration_seconds",
    ]
    assert all(column.nullable is True for _, column in recorder.added)

    migration.downgrade()
    assert recorder.dropped == [
        ("practice_sheets", "generation_duration_seconds"),
        ("practice_sheets", "generation_started_at"),
    ]


def test_practice_sheet_model_exposes_nullable_generation_timing_columns():
    from app.models.practice_sheet import PracticeSheet

    columns = PracticeSheet.__table__.c
    assert columns.generation_started_at.nullable is True
    assert columns.generation_duration_seconds.nullable is True


def test_sheet_generation_schema_reports_original_question_progress():
    from app.schemas.sheet import SheetGenerationOut, SheetOut

    now = datetime(2026, 8, 17, 12, 0, 0)
    sheet_id = uuid4()
    generation = SheetGenerationOut(
        id=sheet_id,
        generation_status="processing",
        generation_total=57,
        generation_completed=12,
        generation_error_code=None,
        generation_error_message=None,
        generation_duration_seconds=None,
        pdf_url=None,
        updated_at=now,
    )
    sheet = SheetOut(
        id=sheet_id,
        title="错题重练",
        config_json={"derived_per_original": 2},
        pdf_url=None,
        created_at=now,
        updated_at=now,
        generation_status="pending",
        generation_total=57,
        generation_completed=0,
        generation_duration_seconds=38,
    )

    assert generation.generation_completed == 12
    assert generation.generation_total == 57
    assert sheet.generation_status == "pending"
    assert sheet.generation_duration_seconds == 38


def test_generation_timing_rounds_up_and_resets_for_retry():
    from app.tasks.generate_sheet import (
        _complete_sheet_generation_timing,
        _start_sheet_generation_timing,
    )

    started_at = datetime(2026, 8, 18, 1, 0, 0)
    sheet = type(
        "Sheet",
        (),
        {
            "generation_started_at": started_at - timedelta(minutes=1),
            "generation_duration_seconds": 60,
        },
    )()

    _start_sheet_generation_timing(sheet, started_at)
    assert sheet.generation_started_at == started_at
    assert sheet.generation_duration_seconds is None

    _complete_sheet_generation_timing(
        sheet,
        started_at + timedelta(milliseconds=1),
    )
    assert sheet.generation_duration_seconds == 1


def test_generation_timing_clamps_negative_clock_skew_to_zero():
    from app.tasks.generate_sheet import _generation_duration_seconds

    started_at = datetime(2026, 8, 18, 1, 0, 0)

    assert _generation_duration_seconds(
        started_at,
        started_at + timedelta(seconds=12),
    ) == 12
    assert _generation_duration_seconds(
        started_at,
        started_at - timedelta(seconds=1),
    ) == 0


def test_sheet_generation_soft_limit_is_positive():
    from app.config import settings

    assert settings.SHEET_GENERATION_SOFT_TIME_LIMIT_SECONDS > 0


def test_sheet_derivative_batch_configuration_defaults_and_bounds():
    from pydantic import ValidationError

    from app.config import Settings

    values = Settings(_env_file=None)
    assert values.SHEET_DERIVATIVE_GENERATION_MODE == "serial"
    assert values.SHEET_DERIVATIVE_BATCH_SIZE == 8
    assert values.SHEET_DERIVATIVE_MAX_CONCURRENCY == 3
    assert values.SHEET_DERIVATIVE_RESPONSE_VALIDATION_RETRY_COUNT == 2
    assert values.LLM_REQUEST_TIMEOUT_SECONDS == 60

    with pytest.raises(ValidationError):
        Settings(_env_file=None, SHEET_DERIVATIVE_GENERATION_MODE="other")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, SHEET_DERIVATIVE_BATCH_SIZE=21)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, SHEET_DERIVATIVE_MAX_CONCURRENCY=9)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, SHEET_DERIVATIVE_RESPONSE_VALIDATION_RETRY_COUNT=-1)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LLM_REQUEST_TIMEOUT_SECONDS=0)


def test_worker_receives_documented_batch_generation_settings():
    backend = Path(__file__).parents[2]
    compose = (backend / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (backend / ".env.example").read_text(encoding="utf-8")

    for name in (
        "SHEET_DERIVATIVE_GENERATION_MODE",
        "SHEET_DERIVATIVE_BATCH_SIZE",
        "SHEET_DERIVATIVE_MAX_CONCURRENCY",
        "SHEET_DERIVATIVE_RESPONSE_VALIDATION_RETRY_COUNT",
        "LLM_REQUEST_TIMEOUT_SECONDS",
    ):
        assert name in compose
        assert name in env_example


class _GenerationRepository:
    def __init__(self, request):
        self.request = request
        self.progress = []
        self.completed = None
        self.failed = None
        self.complete_error = None

    def claim(self, _sheet_id):
        return self.request

    def update_progress(self, _sheet_id, completed):
        self.progress.append(completed)

    def complete(self, _sheet_id, request, groups, pdf_url):
        if self.complete_error:
            raise self.complete_error
        self.completed = (request, groups, pdf_url)

    def fail(self, _sheet_id, code, message):
        self.failed = (code, message)


def _generation_request(question_count=2, derived_per_original=2):
    from app.tasks.generate_sheet import SheetGenerationQuestion, SheetGenerationRequest
    from app.services.practice_question import PrintableQuestion

    questions = []
    for index in range(question_count):
        original = PrintableQuestion(
            wrong_question_id=f"question-{index}",
            instruction="计算",
            prompt_text=f"{index} + 1",
            question_type="calculation",
            display_text=f"计算\n{index} + 1\n________________",
            answer=str(index + 1),
        )
        questions.append(
            SheetGenerationQuestion(
                id=f"question-{index}",
                difficulty=2,
                subject="math",
                original=original,
            )
        )
    return SheetGenerationRequest(
        title="错题重练",
        student_name="学生",
        subject="math",
        derived_per_original=derived_per_original,
        difficulty_boost=2,
        questions=questions,
    )


def test_worker_reports_per_original_progress_and_completes_once():
    from app.services.practice_question import PrintableQuestion
    from app.tasks.generate_sheet import execute_sheet_generation

    request = _generation_request()
    repository = _GenerationRepository(request)
    derivative_calls = []

    async def derivative_generator(**kwargs):
        derivative_calls.append(kwargs["original"].wrong_question_id)
        return [
            PrintableQuestion(
                wrong_question_id=kwargs["original"].wrong_question_id,
                instruction="计算",
                prompt_text="2 + 2",
                question_type="calculation",
                display_text="计算\n2 + 2\n________________",
                answer="4",
            )
        ]

    async def batch_generator(**_kwargs):
        raise AssertionError("batch generator must not run in serial mode")

    asyncio.run(
        execute_sheet_generation(
            "sheet-id",
            repository=repository,
            derivative_generator=derivative_generator,
            batch_derivative_generator=batch_generator,
            generation_mode="serial",
            pdf_generator=lambda **_kwargs: "/pdfs/sheet.pdf",
            pdf_remover=lambda _path: None,
        )
    )

    assert derivative_calls == ["question-0", "question-1"]
    assert repository.progress == [1, 2]
    assert repository.completed[2] == "/pdfs/sheet.pdf"
    assert len(repository.completed[1]) == 2
    assert repository.failed is None


def test_worker_skips_both_derivative_generators_for_original_only_batch_mode():
    from app.tasks.generate_sheet import execute_sheet_generation

    repository = _GenerationRepository(
        _generation_request(question_count=3, derived_per_original=0)
    )

    async def fail(**_kwargs):
        raise AssertionError("derivative generator must not run")

    asyncio.run(
        execute_sheet_generation(
            "sheet-id",
            repository=repository,
            derivative_generator=fail,
            batch_derivative_generator=fail,
            generation_mode="batch",
            batch_size=2,
            max_concurrency=2,
            pdf_generator=lambda **_kwargs: "/pdfs/originals.pdf",
            pdf_remover=lambda _path: None,
        )
    )

    assert repository.progress == [1, 2, 3]
    assert [group["derivatives"] for group in repository.completed[1]] == [[], [], []]
    assert repository.failed is None


def test_worker_batches_questions_and_restores_order_after_out_of_order_completion():
    from app.services.derivative import DerivativeBatchGenerationResult
    from app.services.practice_question import PrintableQuestion
    from app.tasks.generate_sheet import execute_sheet_generation

    repository = _GenerationRepository(_generation_request(question_count=5))
    calls = []

    async def batch_generator(items, **_kwargs):
        source_ids = [item.source_id for item in items]
        calls.append(source_ids)
        await asyncio.sleep({"question-0": 0.03, "question-2": 0.02}.get(source_ids[0], 0))
        return DerivativeBatchGenerationResult(
            variants_by_source_id={
                item.source_id: [
                    PrintableQuestion(
                        wrong_question_id=item.source_id,
                        instruction="计算",
                        prompt_text=f"derived-{item.source_id}",
                        question_type="calculation",
                        display_text=f"derived-{item.source_id}",
                        answer="1",
                    )
                ]
                for item in items
            },
            usage={"total_tokens": len(items) * 10},
        )

    async def serial_generator(**_kwargs):
        raise AssertionError("serial generator must not run in batch mode")

    asyncio.run(
        execute_sheet_generation(
            "sheet-id",
            repository=repository,
            derivative_generator=serial_generator,
            batch_derivative_generator=batch_generator,
            generation_mode="batch",
            batch_size=2,
            max_concurrency=2,
            pdf_generator=lambda **_kwargs: "/pdfs/batch.pdf",
            pdf_remover=lambda _path: None,
        )
    )

    assert calls == [
        ["question-0", "question-1"],
        ["question-2", "question-3"],
        ["question-4"],
    ]
    groups = repository.completed[1]
    assert [group["original"]["wrong_question_id"] for group in groups] == [
        f"question-{index}" for index in range(5)
    ]
    assert [group["derivatives"][0]["prompt_text"] for group in groups] == [
        f"derived-question-{index}" for index in range(5)
    ]
    assert repository.progress == sorted(repository.progress)
    assert repository.progress[-1] == 5


def test_worker_limits_batch_concurrency():
    from app.services.derivative import DerivativeBatchGenerationResult
    from app.tasks.generate_sheet import execute_sheet_generation

    repository = _GenerationRepository(_generation_request(question_count=6))
    active = 0
    peak_active = 0

    async def batch_generator(items, **_kwargs):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return DerivativeBatchGenerationResult(
            variants_by_source_id={item.source_id: [] for item in items},
            usage={},
        )

    asyncio.run(
        execute_sheet_generation(
            "sheet-id",
            repository=repository,
            batch_derivative_generator=batch_generator,
            generation_mode="batch",
            batch_size=1,
            max_concurrency=2,
            pdf_generator=lambda **_kwargs: "/pdfs/batch.pdf",
            pdf_remover=lambda _path: None,
        )
    )

    assert peak_active == 2


def test_worker_cancels_sibling_batches_after_first_failure():
    from app.services.derivative import DerivativeGenerationError
    from app.tasks.generate_sheet import execute_sheet_generation

    repository = _GenerationRepository(_generation_request(question_count=2))
    first_started = asyncio.Event()
    cancelled = []

    async def batch_generator(items, **_kwargs):
        if items[0].source_id == "question-0":
            first_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.append("question-0")
                raise
        await first_started.wait()
        raise DerivativeGenerationError("provider batch failed")

    asyncio.run(
        execute_sheet_generation(
            "sheet-id",
            repository=repository,
            batch_derivative_generator=batch_generator,
            generation_mode="batch",
            batch_size=1,
            max_concurrency=2,
            pdf_generator=lambda **_kwargs: "/pdfs/never.pdf",
            pdf_remover=lambda _path: None,
        )
    )

    assert cancelled == ["question-0"]
    assert repository.completed is None
    assert repository.failed == (
        "sheet_derivative_failed",
        "衍生题生成失败，请重试或调整为仅原题",
    )


def test_worker_logs_batch_and_phase_timing_without_question_content(caplog):
    from app.services.derivative import DerivativeBatchGenerationResult
    from app.tasks.generate_sheet import execute_sheet_generation

    repository = _GenerationRepository(_generation_request(question_count=2))

    async def batch_generator(items, **_kwargs):
        return DerivativeBatchGenerationResult(
            variants_by_source_id={item.source_id: [] for item in items},
            usage={"prompt_tokens": 20, "completion_tokens": 10},
        )

    with caplog.at_level(logging.INFO):
        asyncio.run(
            execute_sheet_generation(
                "sheet-id",
                repository=repository,
                batch_derivative_generator=batch_generator,
                generation_mode="batch",
                batch_size=2,
                max_concurrency=1,
                pdf_generator=lambda **_kwargs: "/pdfs/batch.pdf",
                pdf_remover=lambda _path: None,
            )
        )

    assert "sheet derivative batch completed" in caplog.text
    assert "sheet derivative phase completed" in caplog.text
    assert "sheet pdf phase completed" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "prompt_tokens" in caplog.text
    assert "0 + 1" not in caplog.text


def test_worker_failure_log_does_not_include_provider_or_question_content(caplog):
    from app.services.derivative import DerivativeGenerationError
    from app.tasks.generate_sheet import execute_sheet_generation

    repository = _GenerationRepository(_generation_request(question_count=1))

    async def batch_generator(**_kwargs):
        try:
            raise ValueError("secret-answer")
        except ValueError as exc:
            raise DerivativeGenerationError("provider-response") from exc

    with caplog.at_level(logging.ERROR):
        asyncio.run(
            execute_sheet_generation(
                "sheet-id",
                repository=repository,
                batch_derivative_generator=batch_generator,
                generation_mode="batch",
                batch_size=1,
                max_concurrency=1,
                pdf_generator=lambda **_kwargs: "/pdfs/never.pdf",
                pdf_remover=lambda _path: None,
            )
        )

    assert "error_type=DerivativeGenerationError" in caplog.text
    assert "cause_types=DerivativeGenerationError->ValueError" in caplog.text
    assert "question_ids=['question-0']" in caplog.text
    assert "secret-answer" not in caplog.text
    assert "provider-response" not in caplog.text
    assert "0 + 1" not in caplog.text


def test_worker_ignores_duplicate_delivery_when_task_cannot_be_claimed():
    from app.tasks.generate_sheet import execute_sheet_generation

    repository = _GenerationRepository(None)
    calls = []

    asyncio.run(
        execute_sheet_generation(
            "sheet-id",
            repository=repository,
            derivative_generator=lambda **_kwargs: calls.append("derivative"),
            pdf_generator=lambda **_kwargs: calls.append("pdf"),
            pdf_remover=lambda _path: calls.append("remove"),
        )
    )

    assert calls == []
    assert repository.completed is None
    assert repository.failed is None


def test_worker_removes_pdf_and_records_safe_failure_when_persistence_fails():
    from app.tasks.generate_sheet import execute_sheet_generation

    repository = _GenerationRepository(_generation_request())
    repository.complete_error = RuntimeError("database password leaked")
    removed = []

    async def derivative_generator(**_kwargs):
        return []

    asyncio.run(
        execute_sheet_generation(
            "sheet-id",
            repository=repository,
            derivative_generator=derivative_generator,
            pdf_generator=lambda **_kwargs: "/pdfs/uncommitted.pdf",
            pdf_remover=removed.append,
        )
    )

    assert removed == ["/pdfs/uncommitted.pdf"]
    assert repository.failed == (
        "sheet_generation_failed",
        "错题集生成失败，请稍后重试",
    )
    assert "password" not in repository.failed[1]


def test_worker_maps_derivative_and_timeout_failures_to_safe_messages():
    from celery.exceptions import SoftTimeLimitExceeded
    from app.services.derivative import DerivativeGenerationError
    from app.tasks.generate_sheet import generation_failure_for

    assert generation_failure_for(
        DerivativeGenerationError("provider response"),
        "derivatives",
    ) == ("sheet_derivative_failed", "衍生题生成失败，请重试或调整为仅原题")
    assert generation_failure_for(
        SoftTimeLimitExceeded(),
        "derivatives",
    ) == ("sheet_generation_timeout", "错题集生成超时，请重试或减少题目数量")


class _Rows:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return self

    def all(self):
        return self.values


class _CreateSheetDb:
    def __init__(self, questions):
        self.questions = questions
        self.added = []
        self.events = []

    async def execute(self, _query):
        return _Rows(self.questions)

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.events.append("commit")

    async def flush(self):
        self.events.append("flush")

    async def refresh(self, value):
        value.id = value.id or uuid4()
        value.created_at = datetime(2026, 8, 17, 12, 0, 0)
        value.updated_at = datetime(2026, 8, 17, 12, 0, 0)
        self.events.append("refresh")


def _question(question_id):
    return type(
        "Question",
        (),
        {
            "id": question_id,
            "difficulty": 2,
            "subject": "math",
            "ocr_raw_json": {
                "instruction": "计算",
                "prompt_text": "1 + 1",
                "question_type": "calculation",
            },
            "ocr_answer": "2",
        },
    )()


def test_create_sheet_commits_pending_record_before_dispatch(monkeypatch):
    from app.api import sheets as sheets_api
    from app.schemas.sheet import SheetCreate

    db = _CreateSheetDb([_question("question-id")])
    dispatched = []

    def enqueue(sheet_id):
        assert db.events == ["commit", "refresh"]
        dispatched.append(sheet_id)

    monkeypatch.setattr(sheets_api, "enqueue_sheet_generation", enqueue, raising=False)

    sheet = asyncio.run(
        sheets_api.create_sheet(
            SheetCreate(question_ids=["question-id"], derived_per_original=0),
            student=type("Student", (), {"id": "student-id", "display_name": "学生"})(),
            db=db,
        )
    )

    assert len(db.added) == 1
    assert sheet.generation_status == "pending"
    assert sheet.generation_total == 1
    assert sheet.generation_completed == 0
    assert dispatched == [str(sheet.id)]


class _OwnedSheetDb:
    def __init__(self, sheet):
        self.sheet = sheet
        self.commits = 0

    async def scalar(self, _query):
        return self.sheet

    async def commit(self):
        self.commits += 1

    async def refresh(self, _value):
        pass


def test_failed_sheet_can_be_reset_and_dispatched_once(monkeypatch):
    from app.api import sheets as sheets_api

    sheet = type(
        "Sheet",
        (),
        {
            "id": uuid4(),
            "generation_status": "failed",
            "generation_completed": 21,
            "generation_error_code": "sheet_derivative_failed",
            "generation_error_message": "衍生题生成失败",
        },
    )()
    db = _OwnedSheetDb(sheet)
    dispatched = []
    monkeypatch.setattr(
        sheets_api,
        "enqueue_sheet_generation",
        lambda sheet_id: dispatched.append(sheet_id),
        raising=False,
    )

    result = asyncio.run(
        sheets_api.retry_sheet_generation(
            str(sheet.id),
            student=type("Student", (), {"id": "student-id"})(),
            db=db,
        )
    )

    assert result.generation_status == "pending"
    assert result.generation_completed == 0
    assert result.generation_error_code is None
    assert dispatched == [str(sheet.id)]
    assert db.commits == 1


@pytest.mark.parametrize("status", ["pending", "processing", "failed"])
def test_unfinished_sheet_cannot_enter_practice_flow(status):
    from app.api.sheets import _require_completed

    sheet = type(
        "Sheet",
        (),
        {
            "generation_status": status,
            "generation_error_message": "衍生题生成失败" if status == "failed" else None,
        },
    )()

    with pytest.raises(HTTPException) as exc_info:
        _require_completed(sheet)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "衍生题生成失败" if status == "failed" else "错题集尚未生成完成"
    )


class _DeleteSheetRouteDb:
    def __init__(self, sheet):
        self.sheet = sheet
        self.events = []

    async def scalar(self, _query):
        return self.sheet

    async def commit(self):
        self.events.append("commit")

    async def rollback(self):
        self.events.append("rollback")


@pytest.mark.parametrize("generation_status", ["pending", "processing"])
def test_active_sheet_cannot_be_deleted(generation_status):
    from app.api.sheets import delete_sheet

    sheet = type(
        "Sheet",
        (),
        {"id": uuid4(), "generation_status": generation_status},
    )()
    db = _DeleteSheetRouteDb(sheet)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            delete_sheet(
                str(sheet.id),
                student=type("Student", (), {"id": "student-id"})(),
                db=db,
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "错题集正在生成，暂不能删除"
    assert db.events == []


@pytest.mark.parametrize("generation_status", ["completed", "failed"])
def test_terminal_sheet_delete_commits_before_immediate_pdf_cleanup(
    generation_status,
    monkeypatch,
):
    from app.api import sheets as sheets_api

    sheet = type(
        "Sheet",
        (),
        {"id": uuid4(), "generation_status": generation_status},
    )()
    db = _DeleteSheetRouteDb(sheet)
    cleanup_job_id = uuid4()

    async def delete_data(received_db, received_sheet, student_id):
        assert received_db is db
        assert received_sheet is sheet
        assert student_id == "student-id"
        db.events.append("delete")
        return cleanup_job_id

    async def cleanup(received_db, received_job_id):
        assert received_db is db
        assert received_job_id == cleanup_job_id
        assert db.events == ["delete", "commit"]
        db.events.append("cleanup_failed_but_queued")
        return False

    monkeypatch.setattr(sheets_api, "delete_sheet_data", delete_data, raising=False)
    monkeypatch.setattr(sheets_api, "attempt_file_cleanup", cleanup, raising=False)

    response = asyncio.run(
        sheets_api.delete_sheet(
            str(sheet.id),
            student=type("Student", (), {"id": "student-id"})(),
            db=db,
        )
    )

    assert response.status_code == 204
    assert db.events == ["delete", "commit", "cleanup_failed_but_queued"]


def test_missing_sheet_delete_returns_not_found():
    from app.api.sheets import delete_sheet

    db = _DeleteSheetRouteDb(None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            delete_sheet(
                str(uuid4()),
                student=type("Student", (), {"id": "student-id"})(),
                db=db,
            )
        )

    assert exc_info.value.status_code == 404


def test_unknown_sheet_status_cannot_be_deleted(monkeypatch):
    from app.api import sheets as sheets_api

    sheet = type(
        "Sheet",
        (),
        {"id": uuid4(), "generation_status": "archived"},
    )()
    db = _DeleteSheetRouteDb(sheet)

    async def unexpected_delete(*_args, **_kwargs):
        raise AssertionError("unknown status must not enter deletion")

    monkeypatch.setattr(
        sheets_api,
        "delete_sheet_data",
        unexpected_delete,
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            sheets_api.delete_sheet(
                str(sheet.id),
                student=type("Student", (), {"id": "student-id"})(),
                db=db,
            )
        )

    assert exc_info.value.status_code == 409
    assert db.events == []


def test_cleanup_database_error_after_delete_commit_still_returns_success(
    monkeypatch,
):
    from sqlalchemy.exc import SQLAlchemyError

    from app.api import sheets as sheets_api

    sheet = type(
        "Sheet",
        (),
        {"id": uuid4(), "generation_status": "completed"},
    )()
    db = _DeleteSheetRouteDb(sheet)
    cleanup_job_id = uuid4()

    async def delete_data(*_args, **_kwargs):
        db.events.append("delete")
        return cleanup_job_id

    async def cleanup(*_args, **_kwargs):
        db.events.append("cleanup")
        raise SQLAlchemyError("cleanup database unavailable")

    monkeypatch.setattr(sheets_api, "delete_sheet_data", delete_data, raising=False)
    monkeypatch.setattr(sheets_api, "attempt_file_cleanup", cleanup, raising=False)

    response = asyncio.run(
        sheets_api.delete_sheet(
            str(sheet.id),
            student=type("Student", (), {"id": "student-id"})(),
            db=db,
        )
    )

    assert response.status_code == 204
    assert db.events == ["delete", "commit", "cleanup", "rollback"]
