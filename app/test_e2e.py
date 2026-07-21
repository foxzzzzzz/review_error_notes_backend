"""End-to-end test: upload image → OCR → LLM → sheet → PDF. Run: python app/test_e2e.py"""
import httpx, time, json, uuid

BASE = "http://localhost:8000"
TEST_USER = f"e2e_{uuid.uuid4().hex[:8]}"

def step(label, fn):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    try:
        return fn()
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return None

# ── Step 1: Login ──
token = step("Step 1: Dev login", lambda: httpx.post(
    f"{BASE}/api/auth/dev-login",
    json={"code": TEST_USER}, timeout=10
).json()["token"])
if not token: exit(1)
auth = {"Authorization": f"Bearer {token}"}
print(f"  Token: {token[:50]}...")

# ── Step 2: Create test image ──
step("Step 2: Create test image", lambda: (
    __import__('PIL.Image', fromlist=['Image']).Image.new('RGB', (600, 200), 'white')
    .__class__.__init__ or True,
    print("  Creating image...") or
    setattr(
        __import__('PIL.Image', fromlist=['Image']),
        '_img',
        None
    )
))
from PIL import Image, ImageDraw
img = Image.new('RGB', (600, 250), 'white')
d = ImageDraw.Draw(img)
d.text((20, 30), '1. 小明有12个苹果，吃了3个，还剩几个？', fill='black')
d.text((20, 80), '2. 8 + 7 = ?', fill='black')
d.text((20, 130), '3. 小红买了15支铅笔，用了7支，', fill='black')
d.text((20, 155), '   还剩几支？', fill='black')
d.text((20, 200), '4. 25 - 9 = ?', fill='black')
img.save('/app/uploads/test_page.jpg')
print("  ✅ /app/uploads/test_page.jpg")

# ── Step 3: Upload ──
image_id = step("Step 3: Upload image", lambda: httpx.post(
    f"{BASE}/api/upload/image",
    files={"file": ("test_page.jpg", open("/app/uploads/test_page.jpg", "rb"), "image/jpeg")},
    headers=auth, timeout=10
).json()["image_id"])
if not image_id: exit(1)
print(f"  image_id: {image_id}")

# ── Step 4: Wait for OCR + LLM ──
print("\n  Waiting for OCR + LLM (30s)...")
time.sleep(30)

# ── Step 5: Check questions ──
questions = step("Step 5: List questions", lambda: httpx.get(
    f"{BASE}/api/questions", headers=auth, timeout=10
).json())
if not questions:
    print("  ❌ No questions found. OCR may have failed.")
    exit(1)

qids = []
for q in questions:
    qids.append(q["id"])
    print(f"  Q: {q['ocr_text'][:60] if q.get('ocr_text') else '(no ocr)'}")
    print(f"     subject={q.get('subject')} type={q.get('question_type')} "
          f"tags={q.get('tags')} difficulty={q.get('difficulty')}")
    print(f"     status={q.get('status')}")

# ── Step 6: Generate sheet ──
if len(qids) < 2:
    print(f"\n  ❌ Only {len(qids)} questions, need 2+ to generate sheet")
    exit(1)

sheet = step("Step 6: Generate sheet", lambda: httpx.post(
    f"{BASE}/api/sheets",
    json={
        "title": "E2E测试卷",
        "question_ids": qids[:2],
        "derived_per_original": 1,
        "difficulty_boost": 2,
    },
    headers=auth, timeout=120
).json())
if not sheet: exit(1)
print(f"  Sheet ID: {sheet.get('id')}")
print(f"  PDF URL:  {sheet.get('pdf_url')}")

# ── Step 7: Verify PDF exists ──
import os
pdf_path = sheet.get("pdf_url", "")
if pdf_path.startswith("/pdfs/"):
    full_path = os.path.join("/app", pdf_path.lstrip("/"))
    size = os.path.getsize(full_path) if os.path.exists(full_path) else 0
    print(f"  PDF size: {size} bytes")
    if size > 1000:
        print(f"  ✅ PDF generated successfully!")
    else:
        print(f"  ❌ PDF too small or missing")

print(f"\n{'='*50}")
print(f"🎉 End-to-end test complete!")
print(f"   Questions found: {len(questions)}")
print(f"   Sheet generated: {bool(sheet.get('pdf_url'))}")
