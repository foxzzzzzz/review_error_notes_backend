import json

from app.services.tag_normalization import normalize_tags


def _write_config(tmp_path):
    path = tmp_path / "tag-aliases.json"
    path.write_text(
        json.dumps(
            {
                "aliases": {
                    "pinyin": "拼音",
                    "teacher-marked": "老师批改",
                    "word": "词语",
                    "wrong-character": "错别字",
                },
                "question_type_defaults": {
                    "write_pinyin": "拼音",
                    "write_word": "词语",
                    "fill_blank": "填空",
                    "calculation": "计算",
                    "other": "其他",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_normalizes_known_english_aliases_and_preserves_chinese(tmp_path):
    path = _write_config(tmp_path)

    assert normalize_tags(
        ["#pinyin", "teacher-marked", "拼音", "wrong-character"],
        "write_pinyin",
        str(path),
    ) == ["拼音", "老师批改", "错别字"]


def test_drops_unknown_ascii_and_adds_question_type_default(tmp_path):
    path = _write_config(tmp_path)

    assert normalize_tags(["unknown-tag"], "write_word", str(path)) == ["词语"]


def test_preserves_chinese_tags_and_removes_empty_values(tmp_path):
    path = _write_config(tmp_path)

    assert normalize_tags(
        ["  #小数减法  ", "", "减法", "小数减法"],
        "calculation",
        str(path),
    ) == ["小数减法", "减法", "计算"]
