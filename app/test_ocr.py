"""OCR diagnostic: test PaddleOCR on a generated image. Run: python app/test_ocr.py"""
import os, sys, json, time
sys.path.insert(0, "/app")

# Step 1: Check celery worker is running
print("=== Step 1: Check worker logs ===")
print("(Check with: sudo docker-compose logs worker --tail 20)")

# Step 2: Check WrongImage was created
print("\n=== Step 2: Check DB for uploaded images ===")
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, text
from app.config import settings
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion

async def check_db():
    engine = create_async_engine(settings.DATABASE_URL)
    session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session() as db:
        r = await db.execute(select(WrongImage).order_by(WrongImage.created_at.desc()).limit(3))
        images = r.scalars().all()
        print(f"  Recent images: {len(images)}")
        for img in images:
            print(f"    id={str(img.id)[:8]}... status={img.status} q_count={img.question_count}")

        r = await db.execute(select(WrongQuestion).order_by(WrongQuestion.created_at.desc()).limit(5))
        qs = r.scalars().all()
        print(f"  Recent questions: {len(qs)}")
        for q in qs:
            print(f"    id={str(q.id)[:8]}... status={q.status} ocr_text={(q.ocr_text or '')[:60]}")

asyncio.run(check_db())

# Step 3: Test PaddleOCR directly
print("\n=== Step 3: Test PaddleOCR on test image ===")
img_path = "/app/uploads/test_page.jpg"
if not os.path.exists(img_path):
    print(f"  ❌ Image not found: {img_path}")
    sys.exit(1)

try:
    from app.services.ocr import recognize_text
    result = recognize_text(img_path)
    lines = result.get("lines", [])
    print(f"  Lines detected: {len(lines)}")
    for l in lines[:10]:
        print(f"    text='{l['text']}' conf={l.get('confidence', 0):.2f}")
    if not lines:
        print("  ⚠ No text detected! Test image may have rendering issues.")
        print("  Trying with larger font...")
        from PIL import Image, ImageDraw, ImageFont
        img2 = Image.new('RGB', (800, 300), 'white')
        d = ImageDraw.Draw(img2)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
        d.text((20, 30), "A B C D E", fill='black', font=font)
        d.text((20, 80), "1 2 3 4 5", fill='black', font=font)
        img2.save("/app/uploads/test_ocr_simple.jpg")
        print("  Created /app/uploads/test_ocr_simple.jpg, trying OCR...")
        result2 = recognize_text("/app/uploads/test_ocr_simple.jpg")
        lines2 = result2.get("lines", [])
        print(f"  Lines detected: {len(lines2)}")
        for l in lines2[:10]:
            print(f"    text='{l['text']}' conf={l.get('confidence', 0):.2f}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    import traceback; traceback.print_exc()

# Step 4: Test segmentation
print("\n=== Step 4: Test segmentation ===")
if os.path.exists(img_path):
    try:
        result = recognize_text(img_path)
        lines = result.get("lines", [])
        if lines:
            from app.services.segmenter import segment_questions
            regions = segment_questions(img_path, lines)
            print(f"  Regions: {len(regions)}")
            for i, r in enumerate(regions):
                print(f"    Region {i}: {len(r['text_lines'])} lines, text='{r['text'][:60]}'")
        else:
            print("  Skipped: no OCR lines")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
else:
    print(f"  Image not found: {img_path}")

print("\nDone.")
