"""End-to-end OCR + LLM + PDF test. Run: python app/test_e2e.py"""
import httpx, time, json, uuid, os, sys, asyncio
sys.path.insert(0, "/app")

BASE = "http://localhost:8000"

async def main():
    # ── Login ──
    print("=== Login ===")
    user = f"e2e_{uuid.uuid4().hex[:8]}"
    resp = httpx.post(f"{BASE}/api/auth/dev-login", json={"code": user}, timeout=10)
    token = resp.json()["token"]
    auth = {"Authorization": f"Bearer {token}"}
    print(f"  ✅ {user}")

    # ── Create test image ──
    print("\n=== Create test image ===")
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
        if os.path.exists(fp): font = ImageFont.truetype(fp, 20); print(f"  Font: {fp}"); break
    if font is None: font = ImageFont.load_default()
    img = Image.new('RGB', (600, 200), 'white')
    d = ImageDraw.Draw(img)
    for i, text in enumerate(['1. 12 - 3 = ?', '2. 8 + 7 = ?', '3. 25 - 9 = ?', '4. 15 + 6 = ?']):
        d.text((20, 30 + i*30), text, fill='black', font=font)
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

    # ── DB setup ──
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy import select
    from app.config import settings
    from app.models.wrong_image import WrongImage
    from app.models.wrong_question import WrongQuestion
    engine = create_async_engine(settings.DATABASE_URL)
    DBSession = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def db_status():
        async with DBSession() as db:
            img = (await db.execute(select(WrongImage).where(WrongImage.id == image_id))).scalar_one()
            qs = (await db.execute(select(WrongQuestion).where(WrongQuestion.image_id == image_id))).scalars().all()
            return img, qs

    # ── Poll DB every 3s for up to 45s ──
    print("\n=== Polling DB (max 45s) ===")
    for i in range(15):
        await asyncio.sleep(3)
        img, qs = await db_status()
        elapsed = (i + 1) * 3
        print(f"  +{elapsed:2d}s | image_status={img.status} q_count={img.question_count} | questions={len(qs)}")
        for q in qs:
            print(f"         q={str(q.id)[:8]} status={q.status} ocr={(q.ocr_text or '')[:50]} tags={q.tags} diff={q.difficulty}")
        if img.status == "confirmed" and len(qs) > 0:
            break
    else:
        print("  ⚠ Timed out waiting for processing")

    # ── Final check ──
    print("\n=== Final ===")
    img, qs = await db_status()
    confirmed = [q for q in qs if q.status == "confirmed"]
    print(f"  Image: status={img.status} q_count={img.question_count}")
    print(f"  Questions: total={len(qs)} confirmed={len(confirmed)}")

    if len(confirmed) >= 2:
        resp = httpx.post(f"{BASE}/api/sheets", json={
            "title": "E2E测试卷",
            "question_ids": [str(q.id) for q in confirmed[:2]],
            "derived_per_original": 1,
            "difficulty_boost": 2,
        }, headers=auth, timeout=120)
        sheet = resp.json()
        print(f"  Sheet: {sheet.get('id')} pdf={sheet.get('pdf_url', '')}")
        pdf = sheet.get('pdf_url', '')
        if pdf:
            full = os.path.join('/app', pdf.lstrip('/'))
            if os.path.exists(full):
                print(f"  ✅ PDF: {full} ({os.path.getsize(full)} bytes)")
            else:
                print(f"  ❌ Missing: {full}")
    else:
        print("  ❌ Not enough confirmed questions for sheet")
        print("  Check worker logs: docker-compose logs worker --tail 30")

    await engine.dispose()
    print("\nDone.")

asyncio.run(main())
