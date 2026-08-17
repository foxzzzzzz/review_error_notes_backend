"""Celery task for asynchronous derivative and PDF generation."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable

import httpx
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models.practice_sheet import PracticeSheet
from app.models.sheet_item import SheetItem
from app.models.student import Student
from app.models.wrong_question import WrongQuestion
from app.services.derivative import (
    DerivativeBatchInput,
    DerivativeGenerationError,
    generate_derivative_batch,
    generate_derivative_variants,
)
from app.services.pdf_storage import remove_generated_pdf
from app.services.practice_question import (
    MissingPracticePromptError,
    PrintableQuestion,
    build_printable_questions,
    order_questions,
)
from app.tasks.celery_app import celery_app


logger = logging.getLogger(__name__)


class SheetGenerationInputError(RuntimeError):
    """Raised when selected questions are no longer available to the task."""


@dataclass(frozen=True)
class SheetGenerationQuestion:
    id: str
    difficulty: int
    subject: str
    original: PrintableQuestion


@dataclass(frozen=True)
class SheetGenerationRequest:
    title: str
    student_name: str
    subject: str
    derived_per_original: int
    difficulty_boost: int
    questions: list[SheetGenerationQuestion]


class SheetGenerationRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def claim(self, sheet_id: str) -> SheetGenerationRequest | None:
        with self.session_factory() as db:
            sheet = db.scalar(
                select(PracticeSheet)
                .where(PracticeSheet.id == sheet_id)
                .with_for_update()
            )
            if sheet is None or sheet.generation_status != "pending":
                return None

            config = dict(sheet.config_json or {})
            question_ids = [str(item) for item in config.get("question_ids", [])]
            result = db.execute(
                select(WrongQuestion).where(
                    WrongQuestion.id.in_(question_ids),
                    WrongQuestion.student_id == sheet.student_id,
                    WrongQuestion.deleted_at.is_(None),
                    WrongQuestion.collection_status == "collected",
                )
            )
            questions = order_questions(result.scalars().all(), question_ids)
            if not question_ids or len(questions) != len(question_ids):
                raise SheetGenerationInputError("Selected questions are unavailable")
            originals = build_printable_questions(questions)
            student = db.scalar(select(Student).where(Student.id == sheet.student_id))
            if student is None:
                raise SheetGenerationInputError("Student is unavailable")

            sheet.generation_status = "processing"
            sheet.generation_completed = 0
            sheet.generation_error_code = None
            sheet.generation_error_message = None
            db.commit()

            generation_questions = [
                SheetGenerationQuestion(
                    id=str(question.id),
                    difficulty=question.difficulty or 2,
                    subject=question.subject or "math",
                    original=original,
                )
                for question, original in zip(questions, originals)
            ]
            return SheetGenerationRequest(
                title=sheet.title or "错题重练",
                student_name=student.display_name or "学生",
                subject=generation_questions[0].subject,
                derived_per_original=int(config.get("derived_per_original", 0)),
                difficulty_boost=int(config.get("difficulty_boost", 2)),
                questions=generation_questions,
            )

    def update_progress(self, sheet_id: str, completed: int) -> None:
        with self.session_factory() as db:
            sheet = db.scalar(
                select(PracticeSheet)
                .where(PracticeSheet.id == sheet_id)
                .with_for_update()
            )
            if sheet is None or sheet.generation_status != "processing":
                raise RuntimeError("Sheet generation is no longer active")
            sheet.generation_completed = completed
            db.commit()

    def complete(
        self,
        sheet_id: str,
        request: SheetGenerationRequest,
        groups: list[dict],
        pdf_url: str,
    ) -> None:
        with self.session_factory() as db:
            sheet = db.scalar(
                select(PracticeSheet)
                .where(PracticeSheet.id == sheet_id)
                .with_for_update()
            )
            if sheet is None or sheet.generation_status != "processing":
                raise RuntimeError("Sheet generation is no longer active")

            sort_order = 0
            for question, group in zip(request.questions, groups):
                original = group["original"]
                db.add(
                    SheetItem(
                        sheet_id=sheet.id,
                        wrong_question_id=question.id,
                        question_type="original",
                        question_text=original["display_text"],
                        question_snapshot={
                            "question_type": "original",
                            "question_text": original["display_text"],
                            "source_wrong_question_id": question.id,
                            "sort_order": sort_order,
                        },
                        sort_order=sort_order,
                        generation_method="vision",
                    )
                )
                sort_order += 1
                for derivative in group["derivatives"]:
                    db.add(
                        SheetItem(
                            sheet_id=sheet.id,
                            wrong_question_id=question.id,
                            question_type="derived",
                            derived_from=question.id,
                            question_text=derivative["display_text"],
                            question_snapshot={
                                "question_type": "derived",
                                "question_text": derivative["display_text"],
                                "source_wrong_question_id": question.id,
                                "sort_order": sort_order,
                            },
                            sort_order=sort_order,
                            generation_method="llm",
                        )
                    )
                    sort_order += 1

            sheet.pdf_url = pdf_url
            sheet.generation_status = "completed"
            sheet.generation_completed = sheet.generation_total
            sheet.generation_error_code = None
            sheet.generation_error_message = None
            db.commit()

    def fail(self, sheet_id: str, code: str, message: str) -> None:
        with self.session_factory() as db:
            sheet = db.scalar(
                select(PracticeSheet)
                .where(PracticeSheet.id == sheet_id)
                .with_for_update()
            )
            if sheet is None or sheet.generation_status == "completed":
                return
            sheet.generation_status = "failed"
            sheet.generation_error_code = code
            sheet.generation_error_message = message
            db.commit()


def generation_failure_for(error: Exception, phase: str) -> tuple[str, str]:
    if isinstance(error, SoftTimeLimitExceeded):
        return "sheet_generation_timeout", "错题集生成超时，请重试或减少题目数量"
    if isinstance(error, DerivativeGenerationError):
        return "sheet_derivative_failed", "衍生题生成失败，请重试或调整为仅原题"
    if isinstance(error, (MissingPracticePromptError, SheetGenerationInputError)):
        return "sheet_questions_unavailable", "部分错题已不可用，请重新选择后生成"
    if phase == "pdf":
        return "sheet_pdf_failed", "错题集 PDF 生成失败，请稍后重试"
    return "sheet_generation_failed", "错题集生成失败，请稍后重试"


def _question_group(question: SheetGenerationQuestion, derivatives) -> dict:
    return {
        "original": question.original.model_dump(exclude={"answer"}),
        "derivatives": [
            item.model_dump(exclude={"answer"}) for item in derivatives
        ],
    }


async def _generate_serial_groups(
    sheet_id: str,
    request: SheetGenerationRequest,
    repository,
    derivative_generator: Callable,
) -> list[dict]:
    groups = []
    for completed, question in enumerate(request.questions, start=1):
        target_difficulty = min(
            5,
            question.difficulty + request.difficulty_boost,
        )
        derivatives = await derivative_generator(
            original=question.original,
            difficulty=question.difficulty,
            target_difficulty=target_difficulty,
            subject=question.subject,
            count=request.derived_per_original,
        )
        groups.append(_question_group(question, derivatives))
        repository.update_progress(sheet_id, completed)
    return groups


def _generate_original_only_groups(
    sheet_id: str,
    request: SheetGenerationRequest,
    repository,
) -> list[dict]:
    groups = []
    for completed, question in enumerate(request.questions, start=1):
        groups.append(_question_group(question, []))
        repository.update_progress(sheet_id, completed)
    return groups


async def _generate_batch_groups(
    sheet_id: str,
    request: SheetGenerationRequest,
    repository,
    batch_derivative_generator: Callable,
    batch_size: int,
    max_concurrency: int,
) -> list[dict]:
    batches = [
        request.questions[index : index + batch_size]
        for index in range(0, len(request.questions), batch_size)
    ]
    semaphore = asyncio.Semaphore(max_concurrency)
    results = {}

    async with httpx.AsyncClient(
        timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS
    ) as client:
        async def run_batch(batch_index, questions):
            async with semaphore:
                started_at = time.perf_counter()
                logger.info(
                    "sheet derivative batch started sheet_id=%s batch_index=%s question_count=%s",
                    sheet_id,
                    batch_index,
                    len(questions),
                )
                items = [
                    DerivativeBatchInput(
                        source_id=question.id,
                        original=question.original,
                        difficulty=question.difficulty,
                        target_difficulty=min(
                            5,
                            question.difficulty + request.difficulty_boost,
                        ),
                        subject=question.subject,
                    )
                    for question in questions
                ]
                try:
                    result = await batch_derivative_generator(
                        items=items,
                        count=request.derived_per_original,
                        client=client,
                    )
                except Exception as exc:
                    logger.error(
                        "sheet derivative batch failed sheet_id=%s batch_index=%s "
                        "question_count=%s duration_ms=%s error_type=%s",
                        sheet_id,
                        batch_index,
                        len(questions),
                        round((time.perf_counter() - started_at) * 1000),
                        type(exc).__name__,
                    )
                    raise
                logger.info(
                    "sheet derivative batch completed sheet_id=%s batch_index=%s "
                    "question_count=%s duration_ms=%s usage=%s",
                    sheet_id,
                    batch_index,
                    len(questions),
                    round((time.perf_counter() - started_at) * 1000),
                    result.usage,
                )
                return batch_index, questions, result

        tasks = [
            asyncio.create_task(run_batch(batch_index, questions))
            for batch_index, questions in enumerate(batches)
        ]
        completed = 0
        try:
            for task in asyncio.as_completed(tasks):
                batch_index, questions, result = await task
                results[batch_index] = (questions, result)
                completed += len(questions)
                repository.update_progress(sheet_id, completed)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    groups = []
    for batch_index in range(len(batches)):
        questions, result = results[batch_index]
        for question in questions:
            groups.append(
                _question_group(
                    question,
                    result.variants_by_source_id[question.id],
                )
            )
    return groups


async def execute_sheet_generation(
    sheet_id: str,
    *,
    repository,
    derivative_generator: Callable = generate_derivative_variants,
    batch_derivative_generator: Callable = generate_derivative_batch,
    generation_mode: str | None = None,
    batch_size: int | None = None,
    max_concurrency: int | None = None,
    pdf_generator: Callable | None = None,
    pdf_remover: Callable = remove_generated_pdf,
) -> None:
    phase = "prepare"
    pdf_url = None
    task_started_at = time.perf_counter()
    try:
        request = repository.claim(sheet_id)
        if request is None:
            return

        phase = "derivatives"
        mode = generation_mode or settings.SHEET_DERIVATIVE_GENERATION_MODE
        resolved_batch_size = (
            settings.SHEET_DERIVATIVE_BATCH_SIZE
            if batch_size is None
            else batch_size
        )
        resolved_max_concurrency = (
            settings.SHEET_DERIVATIVE_MAX_CONCURRENCY
            if max_concurrency is None
            else max_concurrency
        )
        logger.info(
            "sheet generation started sheet_id=%s mode=%s question_count=%s "
            "derived_per_original=%s batch_size=%s max_concurrency=%s",
            sheet_id,
            mode,
            len(request.questions),
            request.derived_per_original,
            resolved_batch_size,
            resolved_max_concurrency,
        )
        derivatives_started_at = time.perf_counter()
        if request.derived_per_original == 0:
            groups = _generate_original_only_groups(sheet_id, request, repository)
        elif mode == "batch":
            groups = await _generate_batch_groups(
                sheet_id,
                request,
                repository,
                batch_derivative_generator,
                resolved_batch_size,
                resolved_max_concurrency,
            )
        else:
            groups = await _generate_serial_groups(
                sheet_id,
                request,
                repository,
                derivative_generator,
            )
        logger.info(
            "sheet derivative phase completed sheet_id=%s mode=%s duration_ms=%s",
            sheet_id,
            mode,
            round((time.perf_counter() - derivatives_started_at) * 1000),
        )
        phase = "pdf"
        if pdf_generator is None:
            from app.services.pdf import generate_sheet_pdf

            pdf_generator = generate_sheet_pdf
        pdf_started_at = time.perf_counter()
        pdf_url = pdf_generator(
            student_name=request.student_name,
            subject=request.subject,
            title=request.title,
            groups=groups,
        )
        logger.info(
            "sheet pdf phase completed sheet_id=%s duration_ms=%s",
            sheet_id,
            round((time.perf_counter() - pdf_started_at) * 1000),
        )
        phase = "persist"
        repository.complete(sheet_id, request, groups, pdf_url)
        logger.info(
            "sheet generation completed sheet_id=%s mode=%s duration_ms=%s",
            sheet_id,
            mode,
            round((time.perf_counter() - task_started_at) * 1000),
        )
    except Exception as exc:
        if pdf_url:
            pdf_remover(pdf_url)
        code, message = generation_failure_for(exc, phase)
        repository.fail(sheet_id, code, message)
        logger.error(
            "sheet generation failed sheet_id=%s phase=%s error_type=%s duration_ms=%s",
            sheet_id,
            phase,
            type(exc).__name__,
            round((time.perf_counter() - task_started_at) * 1000),
        )


@celery_app.task(
    name="app.tasks.generate_sheet.generate_sheet_task",
    soft_time_limit=settings.SHEET_GENERATION_SOFT_TIME_LIMIT_SECONDS,
)
def generate_sheet_task(sheet_id: str) -> None:
    sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(sync_url)
    try:
        repository = SheetGenerationRepository(sessionmaker(bind=engine))
        asyncio.run(execute_sheet_generation(sheet_id, repository=repository))
    finally:
        engine.dispose()
