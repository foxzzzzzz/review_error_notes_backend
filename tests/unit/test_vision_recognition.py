import base64
import io
import json

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError


def _write_image(path, size=(2400, 1200)):
    image = Image.new("RGB", size, "white")
    image.save(path, format="JPEG")


def _valid_payload():
    return {
        "items": [
            {
                "raw_text": "qin tin\n蜻蜓",
                "instruction": "看词语写拼音",
                "prompt_text": "蜻蜓",
                "normalized_text": "qīng tíng\n蜻蜓",
                "answer": "qīng tíng",
                "subject": "chinese",
                "question_type": "write_pinyin",
                "tags": ["拼音"],
                "difficulty": 2,
                "confidence": 0.91,
                "uncertain_segments": [],
            }
        ],
        "error_marks": [
            {
                "mark_id": 0,
                "mark_type": "circle",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "confidence": 0.95,
            }
        ],
        "ignored_text": ["Date:"],
    }


def _error_mark():
    from app.services.vision_recognition import ErrorMark

    return ErrorMark(
        mark_id=0,
        mark_type="circle",
        bbox=[0.1, 0.2, 0.3, 0.4],
        confidence=0.95,
    )


def _mock_stage_client(tmp_path, responses, max_retries=0):
    from app.services.vision_recognition import MiniMaxVisionClient

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"content": json.dumps(responses[len(requests) - 1])},
        )

    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=max_retries,
        max_edge=1200,
        jpeg_quality=90,
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    return client, source, requests


def test_prompt_prioritizes_red_error_marks_without_full_page_fallback():
    from app.services.vision_recognition import RECOGNITION_PROMPT

    assert "红圈" in RECOGNITION_PROMPT
    assert "红叉" in RECOGNITION_PROMPT
    assert "每个被标记的独立作答单元输出一个 item" in RECOGNITION_PROMPT
    assert "禁止在没有可靠红标关联时回退输出整页题目" in RECOGNITION_PROMPT
    assert "红色对勾" in RECOGNITION_PROMPT
    assert "[left, top, right, bottom]" in RECOGNITION_PROMPT


def test_mode_prompts_separate_marked_and_unmarked_behavior():
    from app.services.vision_recognition import recognition_prompt_for

    marked = recognition_prompt_for(
        "marked", "chinese", [[0.1, 0.2, 0.3, 0.4]], None
    )
    unmarked = recognition_prompt_for("unmarked", "chinese", [], None)

    assert "[0.1,0.2,0.3,0.4]" in marked
    assert "禁止输出整页未标记题目" in marked
    assert "输出图片中的所有最小可独立作答单元" in unmarked
    assert "不得把所有输出单元直接称为错题" in unmarked
    assert "本次是有红标作业识别" not in unmarked


def test_recognition_correction_adds_only_the_selected_constraint():
    from app.services.vision_recognition import recognition_correction_instruction

    assert "红色错误标记" in recognition_correction_instruction("missed_errors")
    assert "可靠错误证据" in recognition_correction_instruction("false_positives")
    both = recognition_correction_instruction("both")
    assert "红色错误标记" in both
    assert "可靠错误证据" in both
    assert recognition_correction_instruction(None) == ""


def test_prompt_splits_marked_worksheet_into_smallest_answerable_units():
    from app.services.vision_recognition import RECOGNITION_PROMPT

    assert "最小可独立作答单元" in RECOGNITION_PROMPT
    assert "不能把整道编号大题合并成一个 item" in RECOGNITION_PROMPT
    assert "必须分别输出多个 item" in RECOGNITION_PROMPT
    assert "未标记的兄弟小题" in RECOGNITION_PROMPT
    assert "同一作答单元上的红圈、红叉和纠正笔迹视为同一标记组" in RECOGNITION_PROMPT
    assert "同一行存在多个兄弟小题" in RECOGNITION_PROMPT


def test_prompt_describes_complete_word_group_without_literal_answer_examples():
    from app.services.vision_recognition import RECOGNITION_PROMPT

    assert "完整词语格组" in RECOGNITION_PROMPT
    assert "完整词语优先于红色标记的像素覆盖范围" in RECOGNITION_PROMPT
    assert "各字段必须保持同一范围" in RECOGNITION_PROMPT
    for leaked_example in (
        "prompt_text=课文",
        "raw_text=合做",
        "answer=合作",
        "不得只输出 kè、suàn 或 做",
        "例如“计算”或“hé zuò”",
    ):
        assert leaked_example not in RECOGNITION_PROMPT


def test_prompt_separates_student_answer_from_printable_prompt():
    from app.services.vision_recognition import RECOGNITION_PROMPT

    assert '"instruction"' in RECOGNITION_PROMPT
    assert '"prompt_text"' in RECOGNITION_PROMPT
    assert "不得包含学生作答" in RECOGNITION_PROMPT
    assert "difficulty 必须是 1 到 5 的整数" in RECOGNITION_PROMPT


@pytest.mark.parametrize("missing_field", ["instruction", "prompt_text"])
def test_vision_item_requires_clean_practice_prompt_fields(missing_field):
    from app.services.vision_recognition import VisionItem

    item = _valid_payload()["items"][0]
    item.pop(missing_field)

    with pytest.raises(ValidationError):
        VisionItem(**item)


def test_vision_item_allows_empty_raw_text_for_an_unanswered_question():
    from app.services.vision_recognition import VisionItem

    item = _valid_payload()["items"][0]
    item["raw_text"] = ""

    assert VisionItem(**item).raw_text == ""


def test_recognition_prompt_requires_an_empty_raw_text_for_unanswered_questions():
    from app.services.vision_recognition import RECOGNITION_PROMPT

    assert '"raw_text": ""' in RECOGNITION_PROMPT
    assert "未作答" in RECOGNITION_PROMPT


def test_first_stage_returns_error_marks_but_question_items_have_no_bbox():
    from app.services.vision_recognition import ErrorMark, VisionItem, VisionResult

    item = VisionItem(**_valid_payload()["items"][0])
    mark = ErrorMark(**_valid_payload()["error_marks"][0])
    result = VisionResult(items=[item], error_marks=[mark], ignored_text=[])

    assert "bbox" not in result.items[0].model_dump()
    assert result.error_marks[0].mark_id == 0


def test_recognition_prompt_separates_question_content_from_error_mark_coordinates():
    from app.services.vision_recognition import RECOGNITION_PROMPT

    assert '"error_marks"' in RECOGNITION_PROMPT
    assert "题目 item 不得输出 bbox" in RECOGNITION_PROMPT
    assert "不得预先绑定" in RECOGNITION_PROMPT


def test_three_stage_prompts_keep_responsibilities_isolated():
    from app.services.vision_recognition import (
        CONTENT_RECOGNITION_PROMPT,
        MARK_DETECTION_PROMPT,
        MARK_QUESTION_LOCALIZATION_PROMPT,
    )

    assert "不得识别题目内容" in MARK_DETECTION_PROMPT
    assert "学生作答或正确答案" in MARK_DETECTION_PROMPT
    assert "不得合并红圈和红叉" in MARK_DETECTION_PROMPT
    assert "不得识别学生答案" in MARK_QUESTION_LOCALIZATION_PROMPT
    assert "不得修改 mark_id" in MARK_QUESTION_LOCALIZATION_PROMPT
    assert "不得修改题目坐标" in CONTENT_RECOGNITION_PROMPT


def test_mark_detection_rejects_precombined_cross_circle():
    from app.services.vision_recognition import MarkDetectionResult

    payload = {
        "error_marks": [
            {
                "mark_id": 0,
                "mark_type": "cross_circle",
                "bbox": [0.1, 0.1, 0.4, 0.4],
                "cross_bbox": [0.3, 0.1, 0.4, 0.2],
                "circle_bbox": [0.1, 0.2, 0.35, 0.4],
                "confidence": 0.95,
            }
        ]
    }

    with pytest.raises(ValidationError):
        MarkDetectionResult.model_validate(payload)


def test_stage_three_content_has_stable_mark_id_but_no_bbox():
    from app.services.vision_recognition import ContentRecognitionItem

    item = ContentRecognitionItem(
        mark_id=2,
        **_valid_payload()["items"][0],
    )

    assert item.mark_id == 2
    assert "bbox" not in item.model_dump()


@pytest.mark.parametrize(
    ("model_difficulty", "expected_difficulty"),
    [
        (0, 1),
        (6, 5),
    ],
)
def test_vision_item_clamps_out_of_range_integer_difficulty(
    model_difficulty,
    expected_difficulty,
):
    from app.services.vision_recognition import VisionItem

    item = VisionItem(
        **{
            **_valid_payload()["items"][0],
            "difficulty": model_difficulty,
        }
    )

    assert item.difficulty == expected_difficulty


def test_prepare_image_data_url_normalizes_and_limits_dimensions(tmp_path):
    from app.services.vision_recognition import prepare_image_data_url

    source = tmp_path / "large.jpg"
    _write_image(source)

    data_url = prepare_image_data_url(str(source), max_edge=800, jpeg_quality=88)

    assert data_url.startswith("data:image/jpeg;base64,")
    encoded = data_url.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as prepared:
        assert prepared.mode == "RGB"
        assert max(prepared.size) == 800


def test_client_calls_minimax_vlm_and_parses_structured_content(tmp_path):
    from app.services.vision_recognition import MiniMaxVisionClient

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    captured = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(
            200,
            json={
                "content": "<output>\n```json\n"
                + json.dumps(_valid_payload(), ensure_ascii=False)
                + "\n```\n</output>",
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        )

    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com/",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    result = client.recognize(str(source), subject_hint="chinese")

    request = captured["request"]
    body = json.loads(request.content)
    assert request.url == "https://api.minimaxi.com/v1/coding_plan/vlm"
    assert request.headers["authorization"] == "Bearer secret-token"
    assert request.headers["mm-api-source"] == "Minimax-MCP"
    assert body["image_url"].startswith("data:image/jpeg;base64,")
    assert "chinese" in body["prompt"]
    assert result.items[0].raw_text == "qin tin\n蜻蜓"
    assert result.items[0].normalized_text == "qīng tíng\n蜻蜓"
    assert result.ignored_text == ["Date:"]


def test_client_exposes_three_isolated_stage_operations(tmp_path):
    from app.services.vision_recognition import ErrorMark, MiniMaxVisionClient

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    payloads = []
    responses = [
        {
            "error_marks": [
                {
                    "mark_id": 0,
                    "mark_type": "circle",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "cross_bbox": None,
                    "circle_bbox": None,
                    "confidence": 0.95,
                }
            ]
        },
        {
            "items": [
                {
                    "mark_id": 0,
                    "matched": True,
                    "bbox": [0.05, 0.1, 0.4, 0.5],
                    "confidence": 0.92,
                }
            ]
        },
        {"items": [{"mark_id": 0, **_valid_payload()["items"][0]}]},
    ]

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"content": json.dumps(responses[len(payloads) - 1])},
        )

    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    marks = client.detect_marks(str(source), [[0.1, 0.2, 0.3, 0.4]])
    localized = client.locate_marked_questions(str(source), marks.error_marks)
    content = client.recognize_localized_content(str(source), [0], "chinese")

    assert marks.error_marks[0].mark_type == "circle"
    assert localized.items[0].mark_id == 0
    assert content.items[0].mark_id == 0
    assert "学生作答或正确答案" in payloads[0]["prompt"]
    assert "不得识别学生答案" in payloads[1]["prompt"]
    assert "不得修改题目坐标" in payloads[2]["prompt"]
    assert all(payload["image_url"].startswith("data:image/jpeg;base64,") for payload in payloads)


@pytest.mark.parametrize(
    "response_payload",
    [
        {
            "results": [
                {
                    "mark_id": 0,
                    "matched": True,
                    "bbox": [0.05, 0.1, 0.4, 0.5],
                    "confidence": 0.92,
                }
            ]
        },
        {
            "mark_id": 0,
            "matched": True,
            "bbox": [0.05, 0.1, 0.4, 0.5],
            "confidence": 0.92,
        },
    ],
)
def test_mark_localization_normalizes_known_minimax_response_shapes(
    tmp_path,
    response_payload,
):
    client, source, _requests = _mock_stage_client(tmp_path, [response_payload])

    result = client.locate_marked_questions(
        str(source),
        [_error_mark()],
    )

    assert len(result.items) == 1
    assert result.items[0].mark_id == 0
    assert result.items[0].matched is True


def test_mark_localization_requires_items_root(tmp_path):
    from app.services.vision_recognition import VisionRecognitionError

    client, source, _requests = _mock_stage_client(tmp_path, [{}])

    with pytest.raises(VisionRecognitionError) as exc_info:
        client.locate_marked_questions(str(source), [_error_mark()])

    assert exc_info.value.code == "vision_response_schema_invalid"
    assert exc_info.value.diagnostic["expected_root_key"] == "items"
    assert "items" in exc_info.value.diagnostic["validation_fields"]


def test_stage_prompts_and_schema_retry_name_the_exact_items_contract(tmp_path):
    responses = [
        {"answers": []},
        {
            "items": [
                {
                    "mark_id": 0,
                    "matched": True,
                    "bbox": [0.05, 0.1, 0.4, 0.5],
                    "confidence": 0.92,
                }
            ]
        },
        {"items": [{"mark_id": 0, **_valid_payload()["items"][0]}]},
    ]
    client, source, requests = _mock_stage_client(
        tmp_path, responses, max_retries=1
    )

    localized = client.locate_marked_questions(str(source), [_error_mark()])
    content = client.recognize_localized_content(str(source), [0], "chinese")

    assert localized.items[0].mark_id == 0
    assert content.items[0].mark_id == 0
    assert '严格 JSON：{"items"' in requests[0]["prompt"]
    assert '根字段必须是 "items"' in requests[1]["prompt"]
    assert '严格 JSON：{"items"' in requests[2]["prompt"]
    assert 'subject 只能是 "math"、"chinese"、"english"' in requests[2]["prompt"]
    assert 'question_type 只能是 "write_pinyin"' in requests[2]["prompt"]
    assert "instruction 和 prompt_text 都不得为空" in requests[2]["prompt"]


def test_content_recognition_keeps_valid_items_and_reports_invalid_items(
    tmp_path,
    caplog,
):
    valid_item = {"mark_id": 0, **_valid_payload()["items"][0]}
    invalid_item = {
        "mark_id": 1,
        **_valid_payload()["items"][0],
        "raw_text": "private-student-answer",
        "instruction": "",
        "question_type": "pinyin",
    }
    client, source, _requests = _mock_stage_client(
        tmp_path,
        [{"items": [valid_item, invalid_item]}],
    )
    caplog.set_level("INFO")

    result = client.recognize_localized_content(str(source), [0, 1], "chinese")

    assert [item.mark_id for item in result.items] == [0]
    assert len(result.invalid_item_diagnostics) == 1
    rejected = result.invalid_item_diagnostics[0]
    assert rejected["mark_id"] == 1
    assert {
        (error["field"], error["type"])
        for error in rejected["validation_errors"]
    } == {
        ("instruction", "value_error"),
        ("question_type", "literal_error"),
    }
    assert "vision_content_items_rejected" in caplog.text
    assert "private-student-answer" not in caplog.text


@pytest.mark.parametrize("transient_status", [429, 500, 501, 503, 599])
def test_client_retries_transient_status(tmp_path, transient_status):
    from app.services.vision_recognition import MiniMaxVisionClient

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    attempts = []

    def handler(_request):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(transient_status, json={"error": "temporarily unavailable"})
        return httpx.Response(
            200,
            json={"content": json.dumps(_valid_payload(), ensure_ascii=False), "base_resp": {"status_code": 0}},
        )

    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=1,
        max_edge=1200,
        jpeg_quality=90,
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    result = client.recognize(str(source))

    assert len(attempts) == 2
    assert result.items[0].subject == "chinese"


@pytest.mark.parametrize(
    ("first_content", "expected_error_code"),
    [
        ("", "vision_response_empty"),
        ("not-json-private-content", "vision_response_json_invalid"),
        (
            json.dumps(
                {
                    **_valid_payload(),
                    "items": [
                        {
                            key: value
                            for key, value in _valid_payload()["items"][0].items()
                            if key != "raw_text"
                        }
                    ],
                }
            ),
            "vision_response_schema_invalid",
        ),
    ],
)
def test_client_retries_invalid_model_format_with_correction_prompt(
    tmp_path,
    caplog,
    first_content,
    expected_error_code,
):
    from app.services.vision_recognition import MiniMaxVisionClient

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        content = (
            first_content
            if len(requests) == 1
            else json.dumps(_valid_payload(), ensure_ascii=False)
        )
        return httpx.Response(
            200,
            json={"content": content, "base_resp": {"status_code": 0}},
        )

    caplog.set_level("INFO")
    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=1,
        max_edge=1200,
        jpeg_quality=90,
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    result = client.recognize(str(source))

    assert result.items[0].subject == "chinese"
    assert len(requests) == 2
    assert "格式纠偏" not in requests[0]["prompt"]
    assert "格式纠偏" in requests[1]["prompt"]
    assert (
        f"vision_response_retry operation=recognition "
        f"error_code={expected_error_code} attempt=1 max_retries=1"
    ) in caplog.text
    if first_content:
        assert first_content not in caplog.text


def test_invalid_json_diagnostic_describes_shape_without_response_content(tmp_path):
    from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    private_content = '{"items":'
    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "content": private_content,
                    "base_resp": {"status_code": 0},
                },
            )
        ),
    )

    with pytest.raises(VisionRecognitionError) as raised:
        client.recognize(str(source))

    diagnostic = raised.value.diagnostic
    assert raised.value.code == "vision_response_json_invalid"
    assert diagnostic["response_content_length"] == 9
    assert diagnostic["json_error_position"] == 9
    assert diagnostic["json_error_line"] == 1
    assert diagnostic["json_error_column"] == 10
    assert diagnostic["has_markdown_fence"] is False
    assert diagnostic["has_output_wrapper"] is False
    assert diagnostic["first_non_whitespace_char_type"] == "object_start"
    assert diagnostic["last_non_whitespace_char_type"] == "colon"
    assert diagnostic["likely_truncated"] is True
    assert diagnostic["response_attempt"] == 1
    assert diagnostic["response_max_attempts"] == 1
    assert private_content not in str(diagnostic)


def test_invalid_json_diagnostic_inspects_content_inside_markdown_fence(tmp_path):
    from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    private_content = '```json\n{"items":\n```'
    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"content": private_content, "base_resp": {"status_code": 0}},
            )
        ),
    )

    with pytest.raises(VisionRecognitionError) as raised:
        client.recognize(str(source))

    diagnostic = raised.value.diagnostic
    assert diagnostic["has_markdown_fence"] is True
    assert diagnostic["first_non_whitespace_char_type"] == "object_start"
    assert diagnostic["last_non_whitespace_char_type"] == "colon"
    assert diagnostic["likely_truncated"] is True
    assert private_content not in str(diagnostic)


@pytest.mark.parametrize(
    "content",
    [
        lambda payload: "explanation\n" + json.dumps(payload),
        lambda payload: json.dumps(payload) + "\nfinished",
    ],
)
def test_client_rejects_prose_around_json(tmp_path, content):
    from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))

    def handler(_request):
        return httpx.Response(200, json={"content": content(_valid_payload()), "base_resp": {"status_code": 0}})

    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(VisionRecognitionError):
        client.recognize(str(source))


@pytest.mark.parametrize(
    "payload",
    [
        {**_valid_payload(), "items": []},
        {**_valid_payload(), "items": [{**_valid_payload()["items"][0], "confidence": 1.5}]},
        {**_valid_payload(), "items": [{**_valid_payload()["items"][0], "confidence": "0.9"}]},
        {**_valid_payload(), "items": [{**_valid_payload()["items"][0], "unexpected": "value"}]},
        {
            **_valid_payload(),
            "error_marks": [
                {
                    **_valid_payload()["error_marks"][0],
                    "bbox": [0.9, 0.2, 0.2, 0.8],
                }
            ],
        },
        {
            **_valid_payload(),
            "error_marks": [
                {
                    **_valid_payload()["error_marks"][0],
                    "bbox": [0.1, 0.2, 0.1, 0.4],
                }
            ],
        },
    ],
)
def test_client_rejects_invalid_recognition_contract(tmp_path, payload):
    from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))

    def handler(_request):
        return httpx.Response(200, json={"content": json.dumps(payload), "base_resp": {"status_code": 0}})

    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(VisionRecognitionError):
        client.recognize(str(source))


def test_client_errors_do_not_expose_key_or_image_payload(tmp_path):
    from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))

    def handler(_request):
        return httpx.Response(401, json={"error": "invalid key"})

    client = MiniMaxVisionClient(
        api_key="never-log-this-key",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(VisionRecognitionError) as exc_info:
        client.recognize(str(source))

    message = str(exc_info.value)
    assert "never-log-this-key" not in message
    assert "base64" not in message
    assert message == "识别服务返回异常，请稍后重试"


def test_recognition_error_exposes_only_its_safe_category_and_message():
    from app.services.vision_recognition import VisionRecognitionError

    error = VisionRecognitionError(
        "vision_timeout",
        "识别服务响应超时，请稍后重试",
    )

    assert error.code == "vision_timeout"
    assert error.user_message == "识别服务响应超时，请稍后重试"
    assert str(error) == "识别服务响应超时，请稍后重试"


def test_safe_recognition_diagnostic_excludes_provider_response_content():
    from app.services.vision_recognition import (
        VisionRecognitionError,
        safe_recognition_diagnostic,
    )

    diagnostic = safe_recognition_diagnostic(
        VisionRecognitionError(
            "vision_response_schema_invalid",
            "识别结果格式不完整，请稍后重试",
            diagnostic={
                "operation": "recognition",
                "prepared_width": 1475,
                "prepared_height": 2048,
                "validation_fields": ["items.0.raw_text"],
                "raw_response": "provider-private-response",
            },
        )
    )

    assert diagnostic == {
        "operation": "recognition",
        "prepared_width": 1475,
        "prepared_height": 2048,
        "validation_fields": ["items.0.raw_text"],
    }


@pytest.mark.parametrize(
    ("content", "expected_code", "expected_message"),
    [
        ("", "vision_response_empty", "识别结果为空，请稍后重试"),
        (
            "provider-private-response",
            "vision_response_json_invalid",
            "识别结果格式异常，请稍后重试",
        ),
    ],
)
def test_client_classifies_empty_or_non_json_content_without_exposing_response(
    tmp_path,
    content,
    expected_code,
    expected_message,
):
    from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "content": content,
                    "base_resp": {"status_code": 0},
                },
            )
        ),
    )

    with pytest.raises(VisionRecognitionError) as exc_info:
        client.recognize(str(source))

    assert exc_info.value.code == expected_code
    assert exc_info.value.user_message == expected_message
    assert exc_info.value.diagnostic["operation"] == "recognition"
    if content:
        assert content not in str(exc_info.value)
        assert content not in str(exc_info.value.diagnostic)


def test_client_classifies_schema_mismatch_with_safe_context(tmp_path):
    from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    payload = _valid_payload()
    payload["items"][0].pop("raw_text")
    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"content": json.dumps(payload), "base_resp": {"status_code": 0}},
            )
        ),
    )

    with pytest.raises(VisionRecognitionError) as exc_info:
        client.recognize(str(source))

    assert exc_info.value.code == "vision_response_schema_invalid"
    assert exc_info.value.user_message == "识别结果格式不完整，请稍后重试"
    assert exc_info.value.diagnostic["operation"] == "recognition"
    assert "items.0.raw_text" in exc_info.value.diagnostic["validation_fields"]
    assert "secret-token" not in str(exc_info.value.diagnostic)


@pytest.mark.parametrize(
    ("status_code", "expected_code", "expected_message"),
    [
        (429, "vision_rate_limited", "识别服务繁忙，请稍后重试"),
        (503, "vision_service_unavailable", "识别服务暂时不可用，请稍后重试"),
        (401, "vision_http_rejected", "识别服务返回异常，请稍后重试"),
    ],
)
def test_client_classifies_unsuccessful_minimax_responses(
    tmp_path, status_code, expected_code, expected_message,
):
    from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    client = MiniMaxVisionClient(
        api_key="secret-token",
        api_host="https://api.minimaxi.com",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
        transport=httpx.MockTransport(lambda _request: httpx.Response(status_code)),
    )

    with pytest.raises(VisionRecognitionError) as exc_info:
        client.recognize(str(source))

    assert exc_info.value.code == expected_code
    assert exc_info.value.user_message == expected_message
    assert "secret-token" not in str(exc_info.value)


def test_client_classifies_missing_configuration_without_exposing_details(tmp_path):
    from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError

    source = tmp_path / "question.jpg"
    _write_image(source, (400, 300))
    client = MiniMaxVisionClient(
        api_key="",
        api_host="",
        timeout_seconds=5,
        max_retries=0,
        max_edge=1200,
        jpeg_quality=90,
    )

    with pytest.raises(VisionRecognitionError) as exc_info:
        client.recognize(str(source))

    assert (exc_info.value.code, exc_info.value.user_message) == (
        "vision_not_configured", "识别服务尚未配置，请联系管理员",
    )
