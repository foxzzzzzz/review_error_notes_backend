import ast
from pathlib import Path


def test_question_detail_exposes_structured_review_context():
    source = (Path(__file__).parents[2] / "app" / "schemas" / "question.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    question_out = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QuestionOut"
    )
    fields = {
        node.target.id
        for node in question_out.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert "ocr_answer" in fields
    assert "ocr_raw_json" in fields
