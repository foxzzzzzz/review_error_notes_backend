"""Celery task: process uploaded image through OCR + LLM pipeline.

Uses sync SQLAlchemy because Celery tasks are synchronous. Async + Celery
causes asyncpg connection pool conflicts across asyncio.run() event loops."""
from app.tasks.celery_app import celery_app
from app.services.ocr import recognize_text
from app.services.segmenter import segment_questions
from app.config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion
from sqlalchemy import select


@celery_app.task(bind=True)
def process_image(self, image_id: str, filepath: str):
    """OCR → segment → create WrongQuestions → optional LLM analysis."""
    # Sync engine: strip +asyncpg, use psycopg2. Bind metadata so FK resolution works.
    sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(sync_url)
    from app.models import Base
    Base.metadata.create_all(engine, checkfirst=True)

    with Session(engine) as db:
        # Step 1: OCR
        ocr_result = recognize_text(filepath)

        # Step 2: segment into individual questions
        regions = segment_questions(filepath, ocr_result["lines"])

        # Step 3: update WrongImage
        img = db.scalar(select(WrongImage).where(WrongImage.id == image_id))
        if not img:
            engine.dispose()
            return
        img.question_count = len(regions)
        img.status = "segmented"

        # Step 4: create WrongQuestions from OCR text
        question_ids = []
        for i, region in enumerate(regions):
            q = WrongQuestion(
                student_id=img.student_id,
                image_id=img.id,
                crop_region={"bbox": region["bbox"], "index": i},
                subject=img.subject,
                semester=img.semester,
                grade=img.grade,
                ocr_text=region["text"],
                ocr_raw_json={"lines": region["text_lines"]},
                status="ocr_done",
            )
            db.add(q)
            db.flush()
            question_ids.append(str(q.id))

        db.commit()
    engine.dispose()

    # Step 5: LLM analysis (in separate sync session to avoid cross-loop issues)
    if settings.LLM_API_KEY and question_ids:
        try:
            import asyncio
            engine2 = create_engine(sync_url)
            from app.models import Base as _Base
            _Base.metadata.create_all(engine2, checkfirst=True)
            with Session(engine2) as db2:
                for qid in question_ids:
                    q = db2.scalar(select(WrongQuestion).where(WrongQuestion.id == qid))
                    img = db2.scalar(select(WrongImage).where(WrongImage.id == image_id))
                    if not q or not img:
                        continue
                    try:
                        analysis = asyncio.run(_analyze_async(q.ocr_text or ""))
                        q.subject = analysis.get("subject", img.subject)
                        q.question_type = analysis.get("question_type")
                        q.problem_schema = analysis.get("problem_schema")
                        q.difficulty_params = analysis.get("difficulty_params")
                        q.tags = analysis.get("tags", [])
                        q.difficulty = analysis.get("difficulty", 3)
                        q.status = "confirmed"
                        if not img.subject:
                            img.subject = q.subject
                        img.status = "confirmed"
                    except Exception as e:
                        self.update_state(state="LLM_FAILED", meta={"error": str(e)})
                db2.commit()
            engine2.dispose()
        except Exception as e:
            self.update_state(state="LLM_FAILED", meta={"error": str(e)})


async def _analyze_async(text: str) -> dict:
    """Run LLM analysis (async, called via asyncio.run)."""
    from app.services.llm import analyze_question
    return await analyze_question(text)
