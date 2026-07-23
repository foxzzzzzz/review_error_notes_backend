"""Lifecycle helpers for generated PDF files."""

import os
from typing import Optional

from app.config import settings


def remove_generated_pdf(public_url: str, pdf_dir: Optional[str] = None) -> None:
    """Remove one generated PDF without allowing paths outside the PDF directory."""
    prefix = "/pdfs/"
    if not public_url.startswith(prefix):
        return
    filename = public_url[len(prefix):]
    if not filename.endswith(".pdf") or filename != os.path.basename(filename):
        return
    filepath = os.path.join(pdf_dir or settings.PDF_DIR, filename)
    if os.path.isfile(filepath):
        os.remove(filepath)
