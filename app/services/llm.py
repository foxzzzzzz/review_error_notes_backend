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

DERIVATIVE_PROMPT = """基于以下题目，生成一道难度提升的衍生题，只把衍生题文本放在 <output> 标签内。

原题: {question_text}
原题结构: {problem_schema}
当前难度: {difficulty} / 目标难度: {target_difficulty}
科目: {subject}

要求：
- 数学：增大数值、增加计算步骤、或改变问法（正向→逆向）
- 语文：同知识点，替换字词或调整语境
- 英语：替换词汇、变化时态

输出格式示例：
<output>
小明有48颗糖，第一天吃了15颗，第二天吃了9颗，还剩几颗？
</output>"""

OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)
MARKDOWN_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


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
    question_text: str,
    problem_schema: dict,
    difficulty: int,
    target_difficulty: int,
    subject: str,
) -> str:
    """调用 LLM 生成衍生题"""
    prompt = DERIVATIVE_PROMPT.format(
        question_text=question_text,
        problem_schema=json.dumps(problem_schema, ensure_ascii=False),
        difficulty=difficulty,
        target_difficulty=target_difficulty,
        subject=subject,
    )
    try:
        result = await _call_llm(prompt)
        return result if result.strip() else question_text
    except Exception:
        return question_text


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
    async with httpx.AsyncClient(timeout=60) as client:
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
            raise RuntimeError(f"LLM API error {resp.status_code}: {data}")
        msg = data["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning_content", "")
        return _extract_output(raw)


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
