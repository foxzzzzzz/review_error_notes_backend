import ast
from pathlib import Path


MAIN_FILE = Path(__file__).parents[2] / "app" / "main.py"


def test_upload_and_pdf_directories_are_mounted():
    tree = ast.parse(MAIN_FILE.read_text(encoding="utf-8"))
    mounts = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "mount" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant):
            mounts.append(node.args[0].value)

    assert set(mounts) >= {"/uploads", "/pdfs"}


def test_static_directories_are_created_before_mounting():
    source = MAIN_FILE.read_text(encoding="utf-8")

    for setting_name, mount_path in (("UPLOAD_DIR", "/uploads"), ("PDF_DIR", "/pdfs")):
        mkdir = f"os.makedirs(settings.{setting_name}, exist_ok=True)"
        mount = f'app.mount("{mount_path}"'
        assert mkdir in source
        assert mount in source
        assert source.index(mkdir) < source.index(mount)
