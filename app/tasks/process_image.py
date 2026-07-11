from app.tasks.celery_app import celery_app
from app.services.ocr import recognize_text
from app.services.segmenter import segment_questions
from app.services.llm import analyze_question
from app.config import settings
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion
from sqlalchemy import select

engine = create_async_engine(settings.DATABASE_URL)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@celery_app.task(bind=True)
def process_image(self, image_id: str, filepath: str):
    """异步处理一张图片：OCR → 切题 → 单题识别 → LLM 分析"""
    async def _run():
        async with async_session() as db:
            # Step 1: OCR
            ocr_result = recognize_text(filepath)

            # Step 2: 切题
            regions = segment_questions(filepath, ocr_result["lines"])

            # Step 3: 更新 wrong_image
            result = await db.execute(select(WrongImage).where(WrongImage.id == image_id))
            image = result.scalar_one()
            image.question_count = len(regions)
            image.status = "segmented"

            # Step 4: 逐题创建 wrong_question + LLM 分析
            for i, region in enumerate(regions):
                q = WrongQuestion(
                    student_id=image.student_id,
                    image_id=image.id,
                    crop_region={"bbox": region["bbox"], "index": i},
                    subject=image.subject,  # LLM 分析后再确认
                    semester=image.semester,
                    grade=image.grade,
                    ocr_text=region["text"],
                    ocr_raw_json={"lines": region["text_lines"]},
                    status="ocr_done",
                )
                db.add(q)
                await db.flush()

                # Step 5: LLM 分析（如果配置了 API Key）
                if settings.LLM_API_KEY:
                    try:
                        analysis = await analyze_question(region["text"])
                        q.subject = analysis.get("subject", image.subject)
                        q.question_type = analysis.get("question_type")
                        q.problem_schema = analysis.get("problem_schema")
                        q.difficulty_params = analysis.get("difficulty_params")
                        q.tags = analysis.get("tags", [])
                        q.difficulty = analysis.get("difficulty", 3)
                        q.status = "confirmed"
                        if not image.subject:
                            image.subject = q.subject
                    except Exception as e:
                        self.update_state(state="LLM_FAILED", meta={"error": str(e)})

            await db.commit()

    asyncio.run(_run())
