import ast
from pathlib import Path


API_DIR = Path(__file__).parents[2] / "app" / "api"


def _literal(node):
    return node.value if isinstance(node, ast.Constant) else None


def _routes(filename):
    tree = ast.parse((API_DIR / filename).read_text(encoding="utf-8"))
    prefix = ""
    routes = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if getattr(node.value.func, "id", None) == "APIRouter":
                for keyword in node.value.keywords:
                    if keyword.arg == "prefix":
                        prefix = _literal(keyword.value) or ""

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            method = decorator.func.attr.upper()
            if method not in {"GET", "POST", "PATCH", "DELETE"}:
                continue
            path = _literal(decorator.args[0]) if decorator.args else ""
            routes.add((method, "/api" + prefix + (path or "")))

    return routes


def test_public_api_contract_contains_every_miniprogram_endpoint():
    actual = set()
    for filename in ("auth.py", "upload.py", "questions.py", "sheets.py", "profile.py"):
        actual.update(_routes(filename))

    expected = {
        ("POST", "/api/auth/login"),
        ("POST", "/api/auth/bind-phone"),
        ("POST", "/api/upload/image"),
        ("GET", "/api/questions"),
        ("GET", "/api/questions/{question_id}"),
        ("PATCH", "/api/questions/{question_id}"),
        ("DELETE", "/api/questions/{question_id}"),
        ("POST", "/api/sheets"),
        ("GET", "/api/sheets"),
        ("GET", "/api/profile"),
        ("PATCH", "/api/profile"),
    }

    assert actual >= expected
