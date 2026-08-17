from pathlib import Path

from app.services.pdf_layout import split_trailing_answer_line


BACKEND = Path(__file__).parents[2]
TEMPLATE = BACKEND / "templates" / "sheet.html"
PDF_SERVICE = BACKEND / "app" / "services" / "pdf.py"
DOCKERFILE = BACKEND / "Dockerfile"


def test_split_trailing_answer_line_extracts_terminal_placeholder():
    assert split_trailing_answer_line('根据拼音“lán tiān”写词语：________________') == {
        "prompt": '根据拼音“lán tiān”写词语：',
        "has_answer_line": True,
    }


def test_split_trailing_answer_line_preserves_internal_underscores():
    text = "变量_a 与 b，不含末尾填空线"

    assert split_trailing_answer_line(text) == {
        "prompt": text,
        "has_answer_line": False,
    }


def test_template_is_grouped_without_answer_page():
    source = TEMPLATE.read_text(encoding="utf-8")

    assert "{% for group in groups %}" in source
    assert "original_items" not in source
    assert "derived_items" not in source
    assert "position: fixed" not in source
    assert "break-inside: avoid" in source
    assert "Noto Sans CJK SC" in source
    assert "答案" not in source


def test_template_renders_semantic_answer_lines():
    source = TEMPLATE.read_text(encoding="utf-8")

    assert "split_trailing_answer_line" in source
    assert 'class="question-prompt"' in source
    assert 'class="answer-line"' in source


def test_template_uses_approved_compact_two_column_layout():
    source = TEMPLATE.read_text(encoding="utf-8")

    assert "margin: 12mm 12mm 14mm;" in source
    assert "column-count: 2;" in source
    assert "column-gap: 6mm;" in source
    assert "font-size: 11.5pt;" in source
    assert "font-size: 9.5pt;" in source
    assert "line-height: 1.45;" in source
    assert "padding: 3mm 4mm;" in source
    assert "margin: 0 0 3.5mm;" in source


def test_template_keeps_groups_intact_and_answer_line_flexible():
    source = TEMPLATE.read_text(encoding="utf-8")

    assert "break-inside: avoid-column;" in source
    assert ".question-content" in source
    assert "display: flex;" in source
    assert "flex-wrap: wrap;" in source
    assert ".answer-line" in source
    assert "min-width: 22mm;" in source


def test_pdf_service_accepts_grouped_questions_only():
    source = PDF_SERVICE.read_text(encoding="utf-8")

    assert "groups: list[dict]" in source
    assert "groups=groups" in source
    assert "original_items" not in source
    assert "derived_items" not in source


def test_pdf_service_registers_answer_line_filter():
    source = PDF_SERVICE.read_text(encoding="utf-8")

    assert 'env.filters["split_trailing_answer_line"]' in source


def test_docker_installs_deterministic_chinese_font():
    source = DOCKERFILE.read_text(encoding="utf-8")

    assert "fonts-noto-cjk" in source
