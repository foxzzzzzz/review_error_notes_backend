import ast
from pathlib import Path


MAIN_FILE = Path(__file__).parents[2] / "app" / "main.py"


def test_private_file_directories_are_not_mounted_as_public_static_files():
    tree = ast.parse(MAIN_FILE.read_text(encoding="utf-8"))
    mounts = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "mount" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant):
            mounts.append(node.args[0].value)

    assert not set(mounts).intersection({"/uploads", "/pdfs", "/avatars"})


def test_private_file_directories_are_created_for_authenticated_downloads():
    source = MAIN_FILE.read_text(encoding="utf-8")

    for setting_name, mount_path in (
        ("UPLOAD_DIR", "/uploads"),
        ("PDF_DIR", "/pdfs"),
        ("AVATAR_DIR", "/avatars"),
    ):
        mkdir = f"os.makedirs(settings.{setting_name}, exist_ok=True)"
        assert mkdir in source
