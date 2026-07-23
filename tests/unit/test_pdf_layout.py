from pathlib import Path


BACKEND = Path(__file__).parents[2]
TEMPLATE = BACKEND / "templates" / "sheet.html"
PDF_SERVICE = BACKEND / "app" / "services" / "pdf.py"
DOCKERFILE = BACKEND / "Dockerfile"


def test_template_is_grouped_single_column_without_answer_page():
    source = TEMPLATE.read_text(encoding="utf-8")

    assert "{% for group in groups %}" in source
    assert "original_items" not in source
    assert "derived_items" not in source
    assert "display: flex" not in source
    assert "position: fixed" not in source
    assert "break-inside: avoid" in source
    assert "Noto Sans CJK SC" in source
    assert "答案" not in source


def test_pdf_service_accepts_grouped_questions_only():
    source = PDF_SERVICE.read_text(encoding="utf-8")

    assert "groups: list[dict]" in source
    assert "groups=groups" in source
    assert "original_items" not in source
    assert "derived_items" not in source


def test_docker_installs_deterministic_chinese_font():
    source = DOCKERFILE.read_text(encoding="utf-8")

    assert "fonts-noto-cjk" in source
