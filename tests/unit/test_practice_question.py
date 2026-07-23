from types import SimpleNamespace

import pytest


def _question(**raw_overrides):
    raw = {
        "instruction": "看词语写拼音",
        "prompt_text": "计算",
        "question_type": "write_pinyin",
    }
    raw.update(raw_overrides)
    return SimpleNamespace(
        id="question-id",
        ocr_text="shǔan",
        ocr_answer="jì suàn",
        ocr_raw_json=raw,
    )


def test_builds_write_pinyin_without_student_wrong_answer():
    from app.services.practice_question import build_printable_question

    result = build_printable_question(_question())

    assert result.display_text == "给“计算”写拼音：________________"
    assert result.answer == "jì suàn"
    assert "shǔan" not in result.display_text


def test_builds_write_word_without_printing_answer():
    from app.services.practice_question import build_printable_question

    result = build_printable_question(
        _question(
            instruction="看拼音写词语",
            prompt_text="hé zuò",
            question_type="write_word",
        )
    )

    assert result.display_text == "根据拼音“hé zuò”写词语：________________"
    assert "合作" not in result.display_text


def test_builds_other_question_with_instruction_prompt_and_answer_area():
    from app.services.practice_question import build_printable_question

    result = build_printable_question(
        _question(
            instruction="计算下面各题",
            prompt_text="12 + 8 =",
            question_type="calculation",
        )
    )

    assert result.display_text == "计算下面各题\n12 + 8 =\n________________"


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"instruction": "", "prompt_text": "计算", "question_type": "write_pinyin"},
        {"instruction": "看词语写拼音", "prompt_text": "", "question_type": "write_pinyin"},
    ],
)
def test_rejects_record_without_structured_practice_prompt(raw):
    from app.services.practice_question import (
        MissingPracticePromptError,
        build_printable_question,
    )

    question = _question()
    question.ocr_raw_json = raw

    with pytest.raises(MissingPracticePromptError):
        build_printable_question(question)


def test_batch_builder_reports_all_unprintable_questions():
    from app.services.practice_question import (
        MissingPracticePromptError,
        build_printable_questions,
    )

    invalid_one = _question()
    invalid_one.ocr_raw_json = {}
    invalid_two = _question()
    invalid_two.ocr_raw_json = {"instruction": "", "prompt_text": ""}

    with pytest.raises(MissingPracticePromptError) as exc_info:
        build_printable_questions([_question(), invalid_one, invalid_two])

    assert exc_info.value.count == 2
