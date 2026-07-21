"""End-to-end OCR + LLM + PDF test with diagnostics. Run: python app/test_e2e.py"""
import httpx, time, json, uuid, os, sys, asyncio
sys.path.insert(0, "/app")

BASE = "http://localhost:8000"
TEST_USER = f"e2e_{uuid.uuid4().hex[:8]}"

# ── Login ──
print("=== Login ===")
resp = httpx.post(f"{BASE}/api/auth/dev-login", json={"code": TEST_USER}, timeout=10)
token = resp.json()["token"]
auth = {"Authorization": f"Bearer {token}"}
print(f"  ✅ user={TEST_USER}")

# ── Create test image ──
print("\n=== Create test image ===")
from PIL import Image, ImageDraw, ImageFont
font = None
for fp in [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]:
    if os.path.exists(fp):
        font = ImageFont.truetype(fp, 20)
        print(f"  Font: {fp}")
        break
if font is None:
    font = ImageFont.load_default()
    print("  Font: default (no CJK)")

img = Image.new('RGB', (600, 200), 'white')
d = ImageDraw.Draw(img)
d.text((20, 30), '1. 12 - 3 = ?', fill='black', font=font)
d.text((20, 60), '2. 8 + 7 = ?', fill='black', font=font)
d.text((20, 90), '3. 25 - 9 = ?', fill='black', font=font)
d.text((20, 120), '4. 15 + 6 - 4 = ?', fill='black', font=font)
os.makedirs('/app/uploads', exist_ok=True)
img.save('/app/uploads/test_page.jpg')
print("  ✅ Created")

# ── Upload ──
print("\n=== Upload ===")
resp = httpx.post(f"{BASE}/api/upload/image",
    files={"file": ("test_page.jpg", open("/app/uploads/test_page.jpg", "rb"), "image/jpeg")},
    headers=auth, timeout=10)
image_id = resp.json()["image_id"]
print(f"  image_id: {image_id}")

# ── Check WrongImage status immediately ──
print("\n=== Check DB (immediate) ===")
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select
from app.config import settings
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion

engine = create_async_engine(settings.DATABASE_URL)
DBSession = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def check_db():
    async with DBSession() as db:
        img = (await db.execute(
            select(WrongImage).where(WrongImage.id == image_id)
        )).scalar_one()
        print(f"  Image status: {img.status}  q_count: {img.question_count}")

        qs = (await db.execute(
            select(WrongQuestion).where(WrongQuestion.image_id == image_id)
        )).scalars().all()
        print(f"  Questions: {len(qs)}")
        for q in qs:
            print(f"    id={str(q.id)[:8]} status={q.status} ocr={q.ocr_text or '(none)'}")

asyncio.run(check_db())

# ── Wait and check ──
for delay in [5, 10, 15, 20]:
    time.sleep(5)
    print(f"\n🕐 +{delay}s check:")
    asyncio.run(check_db())

# ── Final: generate sheet if questions exist ──
print("\n=== Final check ===")
async def final():
    async with DBSession() as db:
        qs = (await db.execute(
            select(WrongQuestion).where(WrongQuestion.student_id == (
                (await db.execute(
                    select(WrongImage.student_id).where(WrongImage.id == image_id)
                )).scalar_one()
            ))
        )).scalars().all()
        confirmed = [q for q in qs if q.status == "confirmed"]
        print(f"  Total questions: {len(qs)}, confirmed: {len(confirmed)}")

        if len(confirmed) >= 2:
            print("  ✅ Generating sheet...")
            resp = httpx.post(f"{BASE}/api/sheets", json={
                "title": "E2E测试卷",
                "question_ids": [str(q.id) for q in confirmed[:2]],
                "derived_per_original": 1,
                "difficulty_boost": 2,
            }, headers=auth, timeout=120)
            sheet = resp.json()
            print(f"  Sheet: {sheet.get('id')}")
            pdf = sheet.get('pdf_url', '')
            if pdf:
                full = os.path.join('/app', pdf.lstrip('/'))
                if os.path.exists(full):
                    print(f"  ✅ PDF: {full} ({os.path.getsize(full)} bytes)")
                else:
                    print(f"  ❌ PDF not found: {full}")
        else:
            print("  ❌ Not enough confirmed questions")

asyncio.run(final())
print("\nDone.")
