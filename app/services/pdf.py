import os, uuid
from datetime import datetime
from weasyprint import HTML
from jinja2 import Environment, FileSystemLoader
from app.config import settings

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "templates")
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))


def generate_sheet_pdf(
    student_name: str,
    subject: str,
    title: str,
    original_items: list[str],
    derived_items: list[str],
) -> str:
    """Generate A4 two-column sheet PDF, return file path."""
    template = env.get_template("sheet.html")
    html = template.render(
        student_name=student_name,
        subject=subject_map(subject),
        title=title,
        date=datetime.now().strftime("%Y-%m-%d"),
        original_items=original_items,
        derived_items=derived_items,
    )
    filename = f"{uuid.uuid4()}.pdf"
    filepath = os.path.join(settings.PDF_DIR, filename)
    os.makedirs(settings.PDF_DIR, exist_ok=True)
    HTML(string=html).write_pdf(filepath)
    return f"/pdfs/{filename}"


def subject_map(s: str) -> str:
    return {"math": "数学", "chinese": "语文", "english": "英语"}.get(s, s)
