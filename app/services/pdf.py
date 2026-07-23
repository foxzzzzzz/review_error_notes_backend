from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

from app.config import settings

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "templates")
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)


def generate_sheet_pdf(
    student_name: str,
    subject: str,
    title: str,
    groups: list[dict],
) -> str:
    """Generate a grouped A4 practice sheet and return its public path."""
    template = env.get_template("sheet.html")
    html = template.render(
        student_name=student_name,
        subject=subject_map(subject),
        title=title,
        date=datetime.now().strftime("%Y-%m-%d"),
        groups=groups,
    )
    filename = f"{uuid.uuid4()}.pdf"
    os.makedirs(settings.PDF_DIR, exist_ok=True)
    filepath = os.path.join(settings.PDF_DIR, filename)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=settings.PDF_DIR,
            suffix=".pdf",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
        HTML(string=html).write_pdf(temporary_path)
        os.replace(temporary_path, filepath)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    return f"/pdfs/{filename}"


def subject_map(subject: str) -> str:
    return {
        "math": "数学",
        "chinese": "语文",
        "english": "英语",
    }.get(subject, subject)
