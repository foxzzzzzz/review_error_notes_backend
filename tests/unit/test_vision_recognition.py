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
                "bbox": [0.1, 0.2, 0.4, 0.4],
            }
        ],
        "ignored_text": ["Date:"],
    }


def test_prompt_prioritizes_red_error_marks_and_falls_back_to_all_questions():
    from app.services.vision_recognition import RECOGNITION_PROMPT

    assert "红圈" in RECOGNITION_PROMPT
    assert "红叉" in RECOGNITION_PROMPT
    assert "每个被标记的独立作答单元输出一个 item" in RECOGNITION_PROMPT
    assert "没有发现明确的红色错误标记" in RECOGNITION_PROMPT
    assert "输出图片中的所有最小可独立作答单元" in RECOGNITION_PROMPT
    assert "红色对勾" in RECOGNITION_PROMPT
    assert "[left, top, right, bottom]" in RECOGNITION_PROMPT


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


@pytest.mark.parametrize(
    "bbox",
    [
        [0.0, 0.45, 1.0, 0.88],
        [0.0, 0.85, 1.0, 1.0],
    ],
)
def test_vision_item_accepts_live_minimax_corner_bbox(bbox):
    from app.services.vision_recognition import VisionItem

    item = VisionItem(**{**_valid_payload()["items"][0], "bbox": bbox})

    assert item.bbox == bbox


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
        {"items": []},
        {"items": [{**_valid_payload()["items"][0], "raw_text": ""}]},
        {"items": [{**_valid_payload()["items"][0], "confidence": 1.5}]},
        {"items": [{**_valid_payload()["items"][0], "confidence": "0.9"}]},
        {"items": [{**_valid_payload()["items"][0], "unexpected": "value"}]},
        {"items": [{**_valid_payload()["items"][0], "bbox": [0.9, 0.2, 0.2, 0.8]}]},
        {"items": [{**_valid_payload()["items"][0], "bbox": [0.1, 0.2, 0.1, 0.4]}]},
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
    assert "401" in message
