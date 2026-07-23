def test_remove_generated_pdf_deletes_only_a_pdf_in_configured_directory(tmp_path):
    from app.services.pdf_storage import remove_generated_pdf

    generated = tmp_path / "sheet-id.pdf"
    generated.write_bytes(b"pdf")
    outside = tmp_path.parent / "outside.pdf"
    outside.write_bytes(b"keep")

    remove_generated_pdf("/pdfs/sheet-id.pdf", str(tmp_path))
    remove_generated_pdf("/pdfs/../outside.pdf", str(tmp_path))

    assert not generated.exists()
    assert outside.read_bytes() == b"keep"
