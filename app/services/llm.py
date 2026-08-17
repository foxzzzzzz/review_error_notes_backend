from dataclasses import dataclass

import httpx, json, re
from app.config import settings

ANALYSIS_PROMPT = """分析以下{subject_hint}题目，将结构化结果放在 <output> 标签中，标签内必须是合法JSON。

JSON格式（参考示例，用实际分析结果替换）：
<output>
{{"subject": "math", "question_type": "word_problem", "problem_schema": {{"operation": "subtraction", "operands": [12, 3]}}, "difficulty_params": {{"num_range": [1, 30], "steps": 1}}, "tags": ["减法", "应用题"], "difficulty": 2}}
</output>

题目: {question_text}"""

ANALYSIS_FALLBACK = {
    "subject": None,
    "question_type": None,
    "problem_schema": {},
    "difficulty_params": {},
    "tags": [],
    "difficulty": 3,
}

DERIVATIVE_PROMPT = """基于以下结构化题目，生成一道同知识点、难度递进的衍生题。

原练习要求: {instruction}
原题提示材料: {prompt_text}
原题题型: {question_type}
原题正确答案: {answer}
当前难度: {difficulty} / 目标难度: {target_difficulty}
科目: {subject}

要求：
- 数学：增大数值、增加计算步骤、或改变问法（正向→逆向）
- 语文：同知识点，替换字词或调整语境
- 英语：替换词汇、变化时态
- 不得返回原题复制品，不得包含学生错误答案或老师批改内容
- instruction、prompt_text、answer 都必须非空
- question_type 只能是 write_pinyin、write_word、fill_blank、calculation、other 之一
- 只在 <output> 内输出一个合法 JSON 对象，不要解释

输出格式：
<output>
{{"instruction":"看词语写拼音","prompt_text":"算式","question_type":"write_pinyin","answer":"suàn shì"}}
</output>"""

BATCH_DERIVATIVE_PROMPT = """基于以下结构化原题，为每道原题生成指定数量的同知识点、难度递进衍生题。

要求：
- 严格保留每个 source_id，不得遗漏、增加或重复
- 每个 source_id 的 variants 数量必须严格等于 derived_count
- 数学：增大数值、增加计算步骤、或改变问法（正向→逆向）
- 语文：同知识点，替换字词或调整语境
- 英语：替换词汇、变化时态
- 不得返回原题复制品，不得包含学生错误答案或老师批改内容
- instruction、prompt_text、answer 都必须非空
- question_type 只能是 write_pinyin、write_word、fill_blank、calculation、other 之一
- 只在 <output> 内输出一个合法 JSON 对象，不要解释

输入：
{batch_json}

输出格式：
<output>
{{"items":[{{"source_id":"原样返回输入ID","variants":[{{"instruction":"看词语写拼音","prompt_text":"算式","question_type":"write_pinyin","answer":"suàn shì"}}]}}]}}
</output>"""

OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)
MARKDOWN_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


@dataclass(frozen=True)
class LLMCallResult:
    content: str
    usage: dict[str, int]


async def analyze_question(question_text: str, subject_hint: str = "") -> dict:
    """调用 LLM 分析题目，返回结构化信息"""
    prompt = ANALYSIS_PROMPT.format(
        subject_hint=subject_hint or "自动判断",
        question_text=question_text,
    )
    try:
        result = await _call_llm(prompt)
        return _parse_json(result)
    except Exception:
        return {**ANALYSIS_FALLBACK}


async def generate_derivative(
    original,
    difficulty: int,
    target_difficulty: int,
    subject: str,
) -> dict:
    """调用 LLM 生成一条结构化衍生题。"""
    prompt = DERIVATIVE_PROMPT.format(
        instruction=original.instruction,
        prompt_text=original.prompt_text,
        question_type=original.question_type,
        answer=original.answer or "",
        difficulty=difficulty,
        target_difficulty=target_difficulty,
        subject=subject,
    )
    result = await _call_llm(prompt)
    if not result.strip():
        raise ValueError("LLM returned an empty derivative")
    parsed = _parse_json(result)
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned an invalid derivative")
    return parsed


async def generate_derivative_batch(items, count: int, client=None) -> tuple[dict, dict[str, int]]:
    """Call the LLM once for multiple source questions."""
    batch_json = json.dumps(
        {
            "derived_count": count,
            "items": [
                {
                    "source_id": item.source_id,
                    "instruction": item.original.instruction,
                    "prompt_text": item.original.prompt_text,
                    "question_type": item.original.question_type,
                    "answer": item.original.answer or "",
                    "difficulty": item.difficulty,
                    "target_difficulty": item.target_difficulty,
                    "subject": item.subject,
                }
                for item in items
            ],
        },
        ensure_ascii=False,
    )
    prompt = BATCH_DERIVATIVE_PROMPT.format(batch_json=batch_json)
    result = await _call_llm_with_usage(prompt, client=client)
    parsed = _parse_json(result.content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned an invalid derivative batch")
    return parsed, result.usage


def _extract_output(raw: str) -> str:
    """Extract content between <output> tags. Falls back to stripping markdown fences then raw."""
    if not raw or not raw.strip():
        return ""
    m = OUTPUT_RE.search(raw)
    if m:
        return m.group(1).strip()
    # Fallback: try extracting from markdown code block
    m = MARKDOWN_JSON_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


async def _call_llm(prompt: str) -> str:
    return (await _call_llm_with_usage(prompt)).content


async def _call_llm_with_usage(prompt: str, client=None) -> LLMCallResult:
    import asyncio

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS)
    last_err = None
    try:
        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{settings.LLM_API_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.LLM_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.LLM_MODEL,
                        "messages": [
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 8192,
                    },
                )
                data = resp.json()
                if resp.status_code != 200:
                    raise RuntimeError(f"LLM API error {resp.status_code}")
                msg = data["choices"][0]["message"]
                raw = msg.get("content") or msg.get("reasoning_content", "")
                usage = {
                    str(key): int(value)
                    for key, value in dict(data.get("usage") or {}).items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }
                return LLMCallResult(content=_extract_output(raw), usage=usage)
            except Exception as e:
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
        raise last_err
    finally:
        if owns_client:
            await client.aclose()


def _parse_json(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response")
    raw = raw.strip()
    # Strip markdown fences
    fences = ["```json", "```"]
    for f in fences:
        if raw.startswith(f):
            raw = raw[len(f):].strip()
            break
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return json.loads(raw)
