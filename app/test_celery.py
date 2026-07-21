"""Celery worker diagnostics. Run: python app/test_celery.py"""
import sys
sys.path.insert(0, "/app")

# Check 1: Can we import the task module?
print("=== Step 1: Import process_image module ===")
try:
    from app.tasks.process_image import process_image
    print(f"  ✅ task imported: {process_image.name}")
except Exception as e:
    print(f"  ❌ import FAILED: {e}")
    import traceback; traceback.print_exc()

# Check 2: Is the task in celery registry?
print("\n=== Step 2: Celery task registry ===")
try:
    from app.tasks.celery_app import celery_app
    tasks = list(celery_app.tasks.keys())
    print(f"  Registered tasks ({len(tasks)}):")
    for t in sorted(tasks):
        print(f"    {t}")
    if "app.tasks.process_image.process_image" not in tasks:
        print("  ❌ process_image NOT registered!")
    else:
        print("  ✅ process_image registered")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Check 3: Try sending a test task
print("\n=== Step 3: Send test task ===")
try:
    from app.tasks.celery_app import celery_app
    result = celery_app.send_task("app.tasks.process_image.process_image",
        args=["test-id", "/tmp/nonexistent.jpg"])
    print(f"  Task sent, id: {result.id}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Check 4: Check worker logs for the latest upload
print("\n=== Step 4: Latest worker activity ===")
import subprocess
try:
    r = subprocess.run(
        ["celery", "-A", "app.tasks.celery_app", "inspect", "active"],
        capture_output=True, text=True, timeout=5
    )
    print(f"  Active tasks: {r.stdout[:500] or r.stderr[:500]}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

print("\nDone.")
